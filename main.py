import pygame
import moderngl
import numpy as np
import logic
import gui
import editor
import sys

# --- CONFIGURATION ---
SCREEN_SIZE = (1280, 720)
STATE_EDIT = 0
STATE_RUN = 1

TEST_VERTICES = 100
TEST_EDGES = 50

def draw_ui(surface, sorted_edges, current_idx, edges_in_mst, total_vertices, slider_val, slider_rect, is_paused):
    surface.fill((0,0,0,0))
    pygame.draw.rect(surface, (40, 40, 40), slider_rect)
    pygame.draw.rect(surface, (200, 200, 200), slider_rect, 2)
    fill_width = int(slider_rect.width * slider_val)
    pygame.draw.rect(surface, (0, 200, 100), (slider_rect.x, slider_rect.y, fill_width, slider_rect.height))
    
    font = pygame.font.SysFont("Arial", 14)
    speed_text = "Step-by-Step" if slider_val < 0.05 else f"Auto Speed: {int(slider_val*100)}%"
    surface.blit(font.render(speed_text, True, (255,255,255)), (slider_rect.x, slider_rect.y - 20))

    status_font = pygame.font.SysFont("Consolas", 18)
    target = total_vertices - 1
    status_txt = f"MST Edges: {edges_in_mst} / {target}"
    
    if edges_in_mst >= target:
        status_txt += " [COMPLETE]"
        col = (0, 255, 0)
    elif current_idx >= len(sorted_edges):
        status_txt += " [FAILED]"
        col = (255, 0, 0)
    else:
        col = (255, 255, 0)
        
    surface.blit(status_font.render(status_txt, True, col), (20, 20))

def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    screen = pygame.display.set_mode(SCREEN_SIZE, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("Kruskal Sandbox")

    ctx = moderngl.create_context()
    
    graph_editor = editor.GraphEditor(*SCREEN_SIZE)
    ui_renderer = gui.TextureRenderer(ctx) 
    
    graph_renderer = None
    circle_renderer = None
    
    current_state = STATE_EDIT
    sorted_edges = []
    parent = []
    rank = []
    
    current_edge_idx = 0
    mst_edges_count = 0
    speed_value = 0.0
    slider_rect = pygame.Rect(20, SCREEN_SIZE[1] - 50, 200, 30)
    ui_surface = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
    
    clock = pygame.time.Clock()
    is_paused = True
    frame_count = 0

    while True:
        events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.VIDEORESIZE:
                # Update viewport on resize
                ctx.viewport = (0, 0, event.w, event.h)
                if graph_renderer:
                    graph_renderer.aspect_ratio = event.w / event.h

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    current_state = STATE_EDIT
                    graph_renderer = None; circle_renderer = None
                    pygame.display.set_caption("Edit Mode")

                elif event.key == pygame.K_t and current_state == STATE_EDIT:
                    # LIMIT TEST (CIRCULAR)
                    sorted_edges, sorted_geom, nodes = logic.prepare_data(TEST_VERTICES, TEST_EDGES)
                    n_vertices = len(nodes)
                    parent = np.arange(n_vertices)
                    rank = np.zeros(n_vertices, dtype=np.int32)
                    
                    graph_renderer = gui.GraphRenderer(ctx, len(sorted_edges), sorted_geom)
                    # Set Aspect Ratio
                    w, h = screen.get_size()
                    graph_renderer.aspect_ratio = w / h
                    
                    circle_renderer = gui.CircleRenderer(ctx, nodes)
                    circle_renderer.update_colors(np.arange(n_vertices, dtype=np.int32))
                    
                    current_edge_idx = 0; mst_edges_count = 0
                    current_state = STATE_RUN; speed_value = 0.15

                elif event.key == pygame.K_RETURN and current_state == STATE_EDIT:
                    # NORMAL RUN
                    raw_edges, geom, nodes = graph_editor.export_data()
                    if raw_edges is not None and len(raw_edges) > 0:
                        sorted_indices = np.argsort(raw_edges[:, 2])
                        sorted_edges = raw_edges[sorted_indices]
                        
                        sorted_geom = np.empty_like(geom)
                        for new_i, old_i in enumerate(sorted_indices):
                            sorted_geom[new_i*2] = geom[old_i*2]
                            sorted_geom[new_i*2+1] = geom[old_i*2+1]

                        n_vertices = int(np.max(sorted_edges[:, :2])) + 1
                        parent = np.arange(n_vertices)
                        rank = np.zeros(n_vertices, dtype=np.int32)
                        
                        graph_renderer = gui.GraphRenderer(ctx, len(sorted_edges), sorted_geom)
                        w, h = screen.get_size()
                        graph_renderer.aspect_ratio = w / h

                        circle_renderer = gui.CircleRenderer(ctx, nodes)
                        circle_renderer.update_colors(np.arange(len(nodes), dtype=np.int32))
                        
                        current_edge_idx = 0; mst_edges_count = 0
                        current_state = STATE_RUN; speed_value = 0.0
                    else:
                        print("Empty Graph")

                elif event.key == pygame.K_RIGHT and current_state == STATE_RUN:
                    is_paused = False # Step manual
                    
            if event.type == pygame.MOUSEWHEEL and current_state == STATE_RUN:
                graph_renderer.zoom *= 1.1 if event.y > 0 else 0.9

            if current_state == STATE_EDIT:
                graph_editor.handle_event(event)

        if current_state == STATE_RUN and mouse_buttons[0]:
            if slider_rect.collidepoint(mouse_pos):
                speed_value = max(0.0, min(1.0, (mouse_pos[0] - slider_rect.x) / slider_rect.width))

        ctx.clear(0.05, 0.05, 0.05)

        if current_state == STATE_EDIT:
            surface = graph_editor.draw()
            ui_renderer.update_texture(surface)
            ui_renderer.render()

        elif current_state == STATE_RUN:
            frame_count += 1
            should_step = False
            batch_size = 1
            
            mst_target = len(parent) - 1
            if mst_edges_count < mst_target and current_edge_idx < len(sorted_edges):
                if speed_value < 0.05:
                    if not is_paused:
                        should_step = True; is_paused = True
                else:
                    batch_size = int(1 + (speed_value * 50)**3) 
                    should_step = True

            if should_step:
                processed, added = logic.process_batch(
                    sorted_edges, current_edge_idx, batch_size, 
                    parent, rank, graph_renderer.state_data
                )
                start_byte = current_edge_idx * 2 * 4
                data_slice = graph_renderer.state_data[current_edge_idx*2 : (current_edge_idx+processed)*2]
                graph_renderer.state_vbo.write(data_slice.tobytes(), offset=start_byte)
                current_edge_idx += processed
                mst_edges_count += added 

            if frame_count % 10 == 0:
                 roots = logic.get_all_roots(parent)
                 valid_roots = roots[:len(circle_renderer.colors)]
                 circle_renderer.update_colors(valid_roots)

            graph_renderer.render()
            
            # Dynamic Point Size: Small if many nodes, big if few
            pt_size = 4.0 if len(parent) > 2000 else 12.0
            circle_renderer.render(graph_renderer.current_matrix, point_size=pt_size)
            
            draw_ui(ui_surface, sorted_edges, current_edge_idx, mst_edges_count, len(parent), speed_value, slider_rect, is_paused)
            ui_renderer.update_texture(ui_surface)
            ui_renderer.render()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()