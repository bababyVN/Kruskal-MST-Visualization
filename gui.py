import moderngl
import numpy as np
from pyrr import Matrix44
import pygame
from config import *

class TextureRenderer:
    """Renders the UI Surface (Table/Sliders) overlay."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    v_uv = in_uv;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D u_texture;
                in vec2 v_uv;
                out vec4 f_color;
                void main() {
                    f_color = texture(u_texture, v_uv);
                }
            '''
        )
        vertices = np.array([-1, -1, 0, 1,  1, -1, 1, 1, -1, 1, 0, 0,  1, 1, 1, 0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_uv')
        self.texture = None

    def update_texture(self, surface):
        texture_data = surface.get_view('1')
        if self.texture is None or self.texture.size != surface.get_size():
            self.texture = self.ctx.texture(surface.get_size(), 4, texture_data)
        else:
            self.texture.write(texture_data)
        self.texture.use()

    def render(self):
        self.ctx.enable(moderngl.BLEND)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

class CircleRenderer:
    """Draws vertices using ModernGL points."""
    def __init__(self, ctx, vertices):
        self.ctx = ctx
        self.n_verts = len(vertices)
        
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec3 in_color;
                in float in_status; // 0.0 = Unselected, 1.0 = Selected (MST)
                out vec3 v_color;
                
                uniform mat4 u_transform;
                uniform float u_base_size;
                uniform float u_zoom; 
                
                void main() {
                    gl_Position = u_transform * vec4(in_vert, 0.0, 1.0);
                    
                    float raw_size = u_base_size * u_zoom;
                    
                    if (in_status > 0.5) {
                        float s = raw_size * 2.0;
                        gl_PointSize = clamp(s, 6.0, 60.0);
                        v_color = in_color;
                    } else {
                        gl_PointSize = clamp(raw_size, 2.0, 20.0);
                        v_color = vec3(0.0, 0.5, 1.0); 
                    }
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    vec2 circ = 2.0 * gl_PointCoord - 1.0;
                    if (dot(circ, circ) > 1.0) discard; 
                    f_color = vec4(v_color, 1.0);
                }
            '''
        )
        
        if self.n_verts > 0:
            self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
            self.colors = np.zeros((self.n_verts, 3), dtype='f4')
            self.color_vbo = self.ctx.buffer(self.colors.tobytes(), dynamic=True)
            self.status = np.zeros(self.n_verts, dtype='f4')
            self.status_vbo = self.ctx.buffer(self.status.tobytes(), dynamic=True)
            
            self.vao = self.ctx.vertex_array(self.prog, [
                (self.vbo, '2f', 'in_vert'),
                (self.color_vbo, '3f', 'in_color'),
                (self.status_vbo, '1f', 'in_status')
            ])

            np.random.seed(999) 
            self.palette = np.random.uniform(0.2, 1.0, (self.n_verts + 1, 3)).astype('f4')
        else:
            self.vbo = None
            self.color_vbo = None
            self.status_vbo = None
            self.vao = None
            self.palette = np.zeros((1, 3), dtype='f4')

    def update_state(self, roots, statuses):
        if self.color_vbo and len(roots) > 0:
            new_colors = self.palette[roots]
            self.color_vbo.write(new_colors.tobytes())
            self.status_vbo.write(statuses.astype('f4').tobytes())

    def render(self, transform_matrix, point_size=14.0, zoom=1.0):
        if self.vao:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.ctx.enable(moderngl.BLEND)
            self.prog['u_transform'].write(transform_matrix)
            self.prog['u_base_size'].value = point_size
            self.prog['u_zoom'].value = zoom
            self.vao.render(moderngl.POINTS)

class GraphRenderer:
    def __init__(self, ctx, n_edges, line_geometry):
        self.ctx = ctx
        self.n_edges = n_edges
        
        self.zoom = 1.0
        self.offset = np.array([0.0, 0.0], dtype='f4')
        self.width = float(SCREEN_SIZE[0])
        self.height = float(SCREEN_SIZE[1])
        
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in float in_state;
                out float v_state;
                uniform mat4 u_transform;
                void main() {
                    v_state = in_state;
                    gl_Position = u_transform * vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in float v_state;
                out vec4 f_color;
                uniform float u_show_rejected;
                uniform float u_show_unseen;
                uniform int u_mode; // 0 = Simulation, 1 = Editor
                
                void main() {
                    if (u_mode == 1) {
                        // EDITOR MODE: Solid Lines
                        f_color = vec4(0.4, 0.4, 0.5, 1.0);
                    } else {
                        // SIMULATION MODE
                        if (v_state > 2.9) {
                            f_color = vec4(0.0, 1.0, 0.2, 1.0); // Selecting (Green)
                        } else if (v_state > 1.9) {
                            f_color = vec4(1.0, 0.8, 0.0, 1.0); // MST (Gold)
                        } else if (v_state > 0.9) {
                            if (u_show_rejected < 0.5) discard;
                            f_color = vec4(1.0, 0.0, 0.0, 0.2); 
                        } else {
                            if (u_show_unseen < 0.5) discard;
                            f_color = vec4(0.3, 0.3, 0.3, 0.15); 
                        }
                    }
                }
            '''
        )
        
        if self.n_edges > 0:
            self.vbo = self.ctx.buffer(line_geometry.tobytes())
            self.state_data = np.zeros(n_edges * 2, dtype='f4') 
            self.state_vbo = self.ctx.buffer(self.state_data.tobytes(), dynamic=True)
            self.vao = self.ctx.vertex_array(self.prog, [
                (self.vbo, '2f', 'in_vert'),
                (self.state_vbo, '1f', 'in_state')
            ])
        else:
            self.vbo = None
            self.state_vbo = None
            self.vao = None
            self.state_data = np.array([])

        self.current_matrix = Matrix44.identity(dtype='f4').tobytes()
        
        self.set_visibility(True, True)

    def set_camera(self, zoom, offset, w, h):
        self.zoom = zoom
        self.offset = offset
        self.width = w
        self.height = h
        self.update_camera()

    def set_visibility(self, show_rejected, show_unseen):
        if 'u_show_rejected' in self.prog:
            self.prog['u_show_rejected'].value = 1.0 if show_rejected else 0.0
            self.prog['u_show_unseen'].value = 1.0 if show_unseen else 0.0

    def set_mode(self, mode_id):
        if 'u_mode' in self.prog:
            self.prog['u_mode'].value = mode_id

    def update_camera(self):
        sx = (2.0 * self.zoom) / self.width
        sy = -(2.0 * self.zoom) / self.height
        tx = (2.0 * self.offset[0] / self.width) - 1.0
        ty = 1.0 - (2.0 * self.offset[1] / self.height)
        
        matrix = np.array([
            [sx,  0.0, 0.0, 0.0],
            [0.0, sy,  0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [tx,  ty,  0.0, 1.0]
        ], dtype='f4')
        
        self.current_matrix = matrix.tobytes()
        if 'u_transform' in self.prog:
            self.prog['u_transform'].write(self.current_matrix)

    def update_states(self, start_idx, count, new_states_slice):
        if self.state_vbo:
            offset = start_idx * 2 * 4 
            self.state_vbo.write(new_states_slice.tobytes(), offset=offset)

    def render(self):
        if self.vao:
            self.vao.render(moderngl.LINES, vertices=self.n_edges * 2)

class RuntimeOverlay:
    """Handles the Pygame UI Overlay during the RUN state."""
    def __init__(self):
        self.settings = {
            'scroll': 0, 
            'show_table': True, 
            'show_ids': True, 
            'show_weights': True,
            'show_deleted': True, 
            'show_unseen': True
        }
        self.slider_rect = pygame.Rect(20, SCREEN_SIZE[1] - 50, 200, 30)
        self.interactive_rects = {}
        self.table_hits = [] 
        
        # Fonts
        self.font_main = pygame.font.SysFont("Arial", 14)
        self.font_bold = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 11)
        self.font_mono = pygame.font.SysFont("Consolas", 18)

    def project_point(self, point, width, height, offset, zoom):
        x, y = point[0], point[1]
        screen_x = x * zoom + offset[0]
        screen_y = y * zoom + offset[1]
        return int(screen_x), int(screen_y)

    def handle_resize(self, w, h):
        self.slider_rect.y = h - 50

    def draw(self, surface, context_data):
        surface.fill((0,0,0,0))
        w, h = surface.get_size()
        self.interactive_rects = {} 
        self.table_hits = [] 
        
        edges = context_data['sorted_edges']
        curr_idx = context_data['current_idx']
        nodes = context_data['nodes']
        renderer = context_data['renderer']
        is_paused = context_data.get('is_paused', True) # Get pause state
        
        # --- HIGHLIGHT CURRENT NODES ---
        if curr_idx < len(edges):
            u, v, _ = edges[curr_idx]
            u_pos = self.project_point(nodes[int(u)], w, h, renderer.offset, renderer.zoom)
            v_pos = self.project_point(nodes[int(v)], w, h, renderer.offset, renderer.zoom)
            pygame.draw.circle(surface, COLOR_SELECTING, u_pos, 20, 2)
            pygame.draw.circle(surface, COLOR_SELECTING, v_pos, 20, 2)

        # --- 1. LABELS WITH THRESHOLD CHECK ---
        if context_data['total_vertices'] < TEXT_RENDER_THRESHOLD and len(edges) < TEXT_RENDER_THRESHOLD:
            if self.settings['show_weights']:
                for i, (u, v, weight) in enumerate(edges):
                    should_draw = True
                    if i < curr_idx:
                        status = context_data['state_data'][i*2]
                        if status > 0.9 and status < 1.9 and not self.settings['show_deleted']: should_draw = False
                    else:
                        if not self.settings['show_unseen']: should_draw = False
                    
                    if should_draw and int(u) < len(nodes) and int(v) < len(nodes):
                        s1 = self.project_point(nodes[int(u)], w, h, renderer.offset, renderer.zoom)
                        s2 = self.project_point(nodes[int(v)], w, h, renderer.offset, renderer.zoom)
                        mid_x, mid_y = (s1[0] + s2[0]) // 2, (s1[1] + s2[1]) // 2
                        
                        if 0 <= mid_x <= w and 0 <= mid_y <= h:
                            txt = self.font_small.render(f"{int(weight)}", True, COLOR_PENDING)
                            bg = pygame.Rect(mid_x, mid_y, txt.get_width(), txt.get_height())
                            pygame.draw.rect(surface, (0,0,0, 150), bg)
                            surface.blit(txt, (mid_x, mid_y))

            if self.settings['show_ids']:
                for i, node in enumerate(nodes):
                    sx, sy = self.project_point(node, w, h, renderer.offset, renderer.zoom)
                    if -20 <= sx <= w + 20 and -20 <= sy <= h + 20:
                        lbl = context_data['labels'][i] if i < len(context_data['labels']) else str(i)
                        txt = self.font_bold.render(lbl, True, COLOR_WHITE)
                        rect = txt.get_rect(center=(sx, sy))
                        surface.blit(txt, rect)
        else:
            if self.settings['show_ids'] or self.settings['show_weights']:
                warn_text = "Graph is too big - ID and Weight rendering disabled"
                lbl = self.font_bold.render(warn_text, True, (255, 80, 80))
                bg = pygame.Rect(0, 0, lbl.get_width() + 20, lbl.get_height() + 10)
                bg.center = (w // 2, h - 80)
                pygame.draw.rect(surface, (30, 30, 30), bg, border_radius=5)
                pygame.draw.rect(surface, (255, 80, 80), bg, 1, border_radius=5)
                surface.blit(lbl, lbl.get_rect(center=bg.center))
        
        # --- 2. CONTROLS (Slider + Buttons) ---
        pygame.draw.rect(surface, (40, 40, 40), self.slider_rect)
        pygame.draw.rect(surface, (200, 200, 200), self.slider_rect, 2)
        fill_width = int(self.slider_rect.width * context_data['speed_val'])
        pygame.draw.rect(surface, COLOR_SELECTING, (self.slider_rect.x, self.slider_rect.y, fill_width, self.slider_rect.height))
        
        speed_text = "Step-by-Step" if context_data['speed_val'] < 0.05 else f"Speed: {int(context_data['speed_val']*100)}%"
        surface.blit(self.font_main.render(speed_text, True, COLOR_WHITE), (self.slider_rect.x, self.slider_rect.y - 20))
        
        # PLAY / PAUSE BUTTON
        btn_play = pygame.Rect(self.slider_rect.right + 20, self.slider_rect.y, 40, 30)
        self.interactive_rects['play_pause'] = btn_play
        pygame.draw.rect(surface, (60, 60, 60), btn_play, border_radius=5)
        pygame.draw.rect(surface, (200, 200, 200), btn_play, 1, border_radius=5)
        
        if is_paused:
            # Draw Green Triangle (Play)
            p1 = (btn_play.x + 12, btn_play.y + 6)
            p2 = (btn_play.x + 12, btn_play.y + 24)
            p3 = (btn_play.x + 30, btn_play.y + 15)
            pygame.draw.polygon(surface, (0, 255, 100), [p1, p2, p3])
        else:
            # Draw Two Bars (Pause)
            pygame.draw.rect(surface, COLOR_WHITE, (btn_play.x + 11, btn_play.y + 7, 6, 16))
            pygame.draw.rect(surface, COLOR_WHITE, (btn_play.x + 23, btn_play.y + 7, 6, 16))

        # RESET BUTTON
        btn_reset = pygame.Rect(btn_play.right + 10, self.slider_rect.y, 80, 30)
        self.interactive_rects['reset'] = btn_reset
        pygame.draw.rect(surface, (200, 50, 50), btn_reset, border_radius=5)
        rst_txt = self.font_main.render("STOP", True, COLOR_WHITE)
        surface.blit(rst_txt, rst_txt.get_rect(center=btn_reset.center))

        # --- 3. STATS ---
        target = context_data['total_vertices'] - 1
        mst_count = context_data['mst_edges_count']
        status_txt = f"MST Edges: {mst_count} / {target}"
        col = (0, 255, 0) if mst_count >= target else (255, 255, 0)
        if curr_idx >= len(edges) and mst_count < target: status_txt += " [FAILED]"; col = (255, 0, 0)
        surface.blit(self.font_mono.render(status_txt, True, col), (20, 15))
        
        weight_txt = f"Total Weight: {context_data['mst_total_weight']:.1f}"
        surface.blit(self.font_mono.render(weight_txt, True, COLOR_PENDING), (20, 35))
        
        # --- TOGGLES ---
        self._draw_checkbox(surface, pygame.Rect(20, 65, 20, 20), "Show IDs", self.settings['show_ids'], 'toggle_ids')
        self._draw_checkbox(surface, pygame.Rect(20, 95, 20, 20), "Show Weights", self.settings['show_weights'], 'toggle_weights')
        self._draw_checkbox(surface, pygame.Rect(160, 65, 20, 20), "Show Rejected (Red)", self.settings['show_deleted'], 'toggle_deleted')
        self._draw_checkbox(surface, pygame.Rect(160, 95, 20, 20), "Show Pending (Grey)", self.settings['show_unseen'], 'toggle_unseen')

        # --- 4. EDGE QUEUE ---
        table_width = 240
        table_x = w - table_width if self.settings['show_table'] else w
        toggle_rect = pygame.Rect(table_x - 30, 10, 30, 30)
        self.interactive_rects['table_toggle'] = toggle_rect
        pygame.draw.rect(surface, (40, 40, 40), toggle_rect, border_top_left_radius=5, border_bottom_left_radius=5)
        arrow_txt = ">" if self.settings['show_table'] else "<"
        surface.blit(self.font_main.render(arrow_txt, True, COLOR_WHITE), (toggle_rect.centerx-5, toggle_rect.centery-8))

        if self.settings['show_table'] and len(edges) > 0:
            pygame.draw.rect(surface, (0, 0, 0, 200), (table_x, 0, table_width, h))
            surface.blit(self.font_mono.render("Edge Queue", True, COLOR_PENDING), (table_x + 10, 10))
            
            row_height = 25
            start_y = 50
            max_rows = (h - start_y) // row_height
            if self.settings['scroll'] > len(edges) - 1: self.settings['scroll'] = 0
            start_idx = self.settings['scroll']
            
            for i in range(max_rows):
                idx = start_idx + i
                if idx >= len(edges): break
                u, v, weight = edges[idx]
                
                lbl_u = context_data['labels'][int(u)] if int(u) < len(context_data['labels']) else str(int(u))
                lbl_v = context_data['labels'][int(v)] if int(v) < len(context_data['labels']) else str(int(v))
                
                bg_color = None
                text_color = (180, 180, 180)
                prefix = "   "
                if idx < curr_idx:
                    status = context_data['state_data'][idx*2]
                    if status > 1.9: bg_color = (255, 215, 0, 80); text_color = (255, 255, 200); prefix = "MST "
                    elif status > 0.9: bg_color = (255, 0, 0, 150); text_color = (255, 200, 200); prefix = "DEL "
                elif idx == curr_idx:
                    bg_color = (0, 255, 100, 50); text_color = COLOR_WHITE; prefix = "-> "
                
                row_rect = pygame.Rect(table_x, start_y + i * row_height, table_width, row_height)
                self.table_hits.append((row_rect, idx))
                
                if bg_color: pygame.draw.rect(surface, bg_color, row_rect)
                mouse_pos = pygame.mouse.get_pos()
                if row_rect.collidepoint(mouse_pos): pygame.draw.rect(surface, (255, 255, 255, 30), row_rect)
                surface.blit(self.font_main.render(f"{prefix}{lbl_u}-{lbl_v} : {weight:.1f}", True, text_color), (table_x + 10, start_y + i * row_height + 5))

    def _draw_checkbox(self, surface, rect, label, state, key):
        pygame.draw.rect(surface, (50, 50, 50), rect)
        pygame.draw.rect(surface, (200, 200, 200), rect, 2)
        if state: pygame.draw.lines(surface, (0, 255, 0), False, [(rect.x+4, rect.y+10), (rect.x+8, rect.y+16), (rect.x+16, rect.y+4)], 2)
        surface.blit(self.font_main.render(label, True, COLOR_WHITE), (rect.right+10, rect.y))
        self.interactive_rects[key] = rect