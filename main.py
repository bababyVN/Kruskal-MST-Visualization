import pygame
import moderngl
import numpy as np
import logic
import gui
import editor
import sys
from config import *

class KruskalApp:
    def __init__(self):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        self.screen = pygame.display.set_mode(SCREEN_SIZE, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption("Kruskal Sandbox")

        # Context & Renderers
        self.ctx = moderngl.create_context()
        self.graph_editor = editor.GraphEditor(*SCREEN_SIZE)
        self.ui_renderer = gui.TextureRenderer(self.ctx)
        self.run_overlay = gui.RuntimeOverlay() # New Overlay Manager
        
        self.graph_renderer = None
        self.circle_renderer = None
        
        # State
        self.current_state = STATE_EDIT
        self.clock = pygame.time.Clock()
        self.ui_surface = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
        
        # Simulation Data
        self.sorted_edges = []
        self.parent = []
        self.rank = []
        self.nodes = []
        self.labels = []
        self.current_edge_idx = 0
        self.mst_edges_count = 0
        self.mst_total_weight = 0.0
        
        # Run Control
        self.speed_value = 0.0
        self.is_paused = True
        self.frame_count = 0
        self.run_zoom = 1.0
        self.run_offset = np.array([0.0, 0.0], dtype='f4')
        self.is_panning = False
        self.pan_start = np.array([0,0], dtype='f4')

    def switch_to_run(self, limit_test=False):
        self.run_zoom = self.graph_editor.zoom
        self.run_offset = self.graph_editor.offset.copy()
        
        # Reset State
        self.run_overlay.settings['scroll'] = 0
        self.mst_total_weight = 0.0 
        
        if limit_test:
            self.run_zoom = 0.5
            self.run_offset = np.array([self.screen.get_width()/2, self.screen.get_height()/2], dtype='f4')
            self.sorted_edges, sorted_geom, raw_nodes = logic.prepare_data(TEST_VERTICES, TEST_EDGES)
            self.nodes = raw_nodes
            self.labels = [str(i) for i in range(len(self.nodes))]
            
            n_vertices = len(self.nodes)
            self.parent = np.arange(n_vertices)
            self.rank = np.zeros(n_vertices, dtype=np.int32)
            
            self.graph_renderer = gui.GraphRenderer(self.ctx, len(self.sorted_edges), sorted_geom)
            self.circle_renderer = gui.CircleRenderer(self.ctx, self.nodes)
            self.circle_renderer.update_state(np.arange(n_vertices), np.zeros(n_vertices))
            
            self.speed_value = 0.15
            self.run_overlay.settings['show_ids'] = False
            self.run_overlay.settings['show_weights'] = False
        else:
            raw_edges, geom, raw_nodes, raw_labels = self.graph_editor.export_data()
            if raw_edges is not None and len(raw_edges) > 0:
                self.nodes = raw_nodes
                self.labels = raw_labels
                
                sorted_indices = np.argsort(raw_edges[:, 2])
                self.sorted_edges = raw_edges[sorted_indices]
                sorted_geom = np.empty_like(geom)
                for new_i, old_i in enumerate(sorted_indices):
                    sorted_geom[new_i*2] = geom[old_i*2]
                    sorted_geom[new_i*2+1] = geom[old_i*2+1]
                
                n_vertices = len(self.nodes)
                self.parent = np.arange(n_vertices)
                self.rank = np.zeros(n_vertices, dtype=np.int32)
                
                self.graph_renderer = gui.GraphRenderer(self.ctx, len(self.sorted_edges), sorted_geom)
                self.circle_renderer = gui.CircleRenderer(self.ctx, self.nodes)
                self.circle_renderer.update_state(np.arange(n_vertices), np.zeros(n_vertices))
                
                self.speed_value = 0.0
                self.run_overlay.settings['show_ids'] = self.graph_editor.show_ids
                self.run_overlay.settings['show_weights'] = self.graph_editor.show_weights
            else:
                print("Empty Graph")
                return

        w, h = self.screen.get_size()
        self.graph_renderer.set_camera(self.run_zoom, self.run_offset, w, h)
        self.current_edge_idx = 0
        self.mst_edges_count = 0
        self.current_state = STATE_RUN

    def handle_events(self):
        events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.VIDEORESIZE:
                self.ctx.viewport = (0, 0, event.w, event.h)
                if self.graph_renderer: 
                    self.graph_renderer.set_camera(self.run_zoom, self.run_offset, event.w, event.h)
                self.ui_surface = pygame.Surface((event.w, event.h), pygame.SRCALPHA)
                self.run_overlay.handle_resize(event.w, event.h)

            if self.current_state == STATE_EDIT:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN: self.switch_to_run()
                    elif event.key == pygame.K_t: self.switch_to_run(limit_test=True)
                
                if self.graph_editor.trigger_run:
                    self.switch_to_run()
                    self.graph_editor.trigger_run = False
                    
                self.graph_editor.handle_event(event)

            elif self.current_state == STATE_RUN:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_ui = False
                    if self.run_overlay.slider_rect.collidepoint(event.pos): clicked_ui = True
                    for r in self.run_overlay.interactive_rects.values():
                        if r.collidepoint(event.pos): clicked_ui = True
                    
                    if event.button == 1:
                        if clicked_ui:
                            rects = self.run_overlay.interactive_rects
                            settings = self.run_overlay.settings
                            if 'reset' in rects and rects['reset'].collidepoint(event.pos): 
                                self.current_state = STATE_EDIT
                            elif 'table_toggle' in rects and rects['table_toggle'].collidepoint(event.pos): 
                                settings['show_table'] = not settings['show_table']
                            elif 'toggle_ids' in rects and rects['toggle_ids'].collidepoint(event.pos): 
                                settings['show_ids'] = not settings['show_ids']
                            elif 'toggle_weights' in rects and rects['toggle_weights'].collidepoint(event.pos): 
                                settings['show_weights'] = not settings['show_weights']
                            elif 'toggle_deleted' in rects and rects['toggle_deleted'].collidepoint(event.pos): 
                                settings['show_deleted'] = not settings['show_deleted']
                            elif 'toggle_unseen' in rects and rects['toggle_unseen'].collidepoint(event.pos): 
                                settings['show_unseen'] = not settings['show_unseen']
                        else:
                            self.is_panning = True
                            self.pan_start = np.array(mouse_pos, dtype='f4')
                    elif event.button == 2:
                        self.is_panning = True
                        self.pan_start = np.array(mouse_pos, dtype='f4')

                elif event.type == pygame.MOUSEBUTTONUP: 
                    self.is_panning = False

                elif event.type == pygame.MOUSEMOTION and self.is_panning:
                    delta = np.array(mouse_pos, dtype='f4') - self.pan_start
                    self.run_offset += delta
                    self.pan_start = np.array(mouse_pos, dtype='f4')
                    self.graph_renderer.set_camera(self.run_zoom, self.run_offset, self.screen.get_width(), self.screen.get_height())

                elif event.type == pygame.MOUSEWHEEL:
                    if self.run_overlay.settings['show_table'] and mouse_pos[0] > self.screen.get_size()[0] - 240:
                        self.run_overlay.settings['scroll'] -= event.y 
                        max_scroll = max(0, len(self.sorted_edges) - 10)
                        self.run_overlay.settings['scroll'] = max(0, min(self.run_overlay.settings['scroll'], max_scroll))
                    else:
                        zoom_factor = 1.1 if event.y > 0 else 0.9
                        m_arr = np.array(mouse_pos, dtype='f4')
                        world_before = (m_arr - self.run_offset) / self.run_zoom
                        self.run_zoom = max(0.0001, self.run_zoom * zoom_factor)
                        self.run_offset = m_arr - (world_before * self.run_zoom)
                        self.graph_renderer.set_camera(self.run_zoom, self.run_offset, self.screen.get_width(), self.screen.get_height())

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: self.current_state = STATE_EDIT
                    elif event.key == pygame.K_LEFT:
                        if self.current_edge_idx > 0:
                            self.current_edge_idx -= 1
                            self.graph_renderer.update_states(self.current_edge_idx, 1, np.array([0.0, 0.0], dtype='f4'))
                            self.graph_renderer.state_data[self.current_edge_idx*2] = 0.0
                            self.graph_renderer.state_data[self.current_edge_idx*2+1] = 0.0
                            
                            # UNDO Logic
                            self.parent = np.arange(len(self.parent))
                            self.rank[:] = 0
                            self.mst_edges_count, self.mst_total_weight = logic.fast_forward_dsu(
                                self.sorted_edges, self.current_edge_idx, self.parent, self.rank
                            )
                            self.is_paused = True
                            # Force update visualization next frame
                            self.force_vis_update = True
                    elif event.key == pygame.K_RIGHT: self.is_paused = False
        
        # Handle Slider Dragging in Run Mode
        if self.current_state == STATE_RUN and mouse_buttons[0] and not self.is_panning:
            if self.run_overlay.slider_rect.collidepoint(mouse_pos):
                self.speed_value = max(0.0, min(1.0, (mouse_pos[0] - self.run_overlay.slider_rect.x) / self.run_overlay.slider_rect.width))

    def update_simulation(self):
        self.frame_count += 1
        should_step = False
        batch_size = 1
        
        # Handle "Force Update" from Undo logic
        if getattr(self, 'force_vis_update', False):
            should_step = False # Don't advance, just visualize
        elif self.mst_edges_count < (len(self.parent)-1) and self.current_edge_idx < len(self.sorted_edges):
            if self.speed_value < 0.05:
                if not self.is_paused: 
                    should_step = True; self.is_paused = True
            else:
                batch_size = int(1 + (self.speed_value * 50)**3)
                should_step = True

        if should_step:
            processed, added, w_added = logic.process_batch(
                self.sorted_edges, self.current_edge_idx, batch_size, 
                self.parent, self.rank, self.graph_renderer.state_data
            )
            
            start_byte = self.current_edge_idx * 2 * 4
            data_slice = self.graph_renderer.state_data[self.current_edge_idx*2 : (self.current_edge_idx+processed)*2]
            self.graph_renderer.state_vbo.write(data_slice.tobytes(), offset=start_byte)
            
            self.current_edge_idx += processed
            self.mst_edges_count += added
            self.mst_total_weight += w_added
            
            if self.run_overlay.settings['show_table']:
                 rows_visible = (self.screen.get_size()[1] - 50) // 25
                 if self.current_edge_idx > self.run_overlay.settings['scroll'] + rows_visible - 2:
                     self.run_overlay.settings['scroll'] = self.current_edge_idx - rows_visible + 2

        # Visual Update Frequency
        if self.frame_count == 1 or self.frame_count % 10 == 0 or should_step or getattr(self, 'force_vis_update', False):
             roots = logic.get_all_roots(self.parent)
             statuses = logic.get_node_statuses(self.parent, self.rank)
             
             valid_roots = roots[:len(self.circle_renderer.colors)]
             valid_stats = statuses[:len(self.circle_renderer.colors)]
             self.circle_renderer.update_state(valid_roots, valid_stats)
             self.force_vis_update = False

        self.graph_renderer.set_visibility(self.run_overlay.settings['show_deleted'], self.run_overlay.settings['show_unseen'])

    def render(self):
        self.ctx.clear(0.05, 0.05, 0.05)

        if self.current_state == STATE_EDIT:
            surface = self.graph_editor.draw()
            self.ui_renderer.update_texture(surface)
            self.ui_renderer.render()

        elif self.current_state == STATE_RUN:
            self.update_simulation()
            
            self.graph_renderer.render()
            pt_size = 5.0 if len(self.parent) > 2000 else 12.0
            self.circle_renderer.render(self.graph_renderer.current_matrix, point_size=pt_size, zoom=self.graph_renderer.zoom)
            
            # Prepare data for overlay
            context = {
                'sorted_edges': self.sorted_edges,
                'current_idx': self.current_edge_idx,
                'mst_edges_count': self.mst_edges_count,
                'total_vertices': len(self.parent),
                'mst_total_weight': self.mst_total_weight,
                'speed_val': self.speed_value,
                'state_data': self.graph_renderer.state_data,
                'nodes': self.nodes,
                'labels': self.labels,
                'renderer': self.graph_renderer
            }
            
            self.run_overlay.draw(self.ui_surface, context)
            self.ui_renderer.update_texture(self.ui_surface)
            self.ui_renderer.render()

        pygame.display.flip()

    def run(self):
        while True:
            self.handle_events()
            self.render()
            self.clock.tick(FPS)

if __name__ == "__main__":
    app = KruskalApp()
    app.run()