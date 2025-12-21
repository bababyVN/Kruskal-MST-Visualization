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

TEST_SPEED = 1
TEST_VERTICES = 1000000
TEST_EDGES = 10000000
TEST_P_Connected = np.exp(
    -np.exp(np.log(TEST_VERTICES) - 2 * TEST_EDGES / TEST_VERTICES)
)
print("Prob of connected graph:",TEST_P_Connected)

def project_point(point, width, height, offset, zoom, aspect_ratio, correct_aspect):
    x, y = point[0], point[1]
    x += offset[0]
    y += offset[1]
    
    if correct_aspect and aspect_ratio > 1.0:
        x *= (zoom / aspect_ratio)
        y *= zoom
    elif correct_aspect:
        x *= zoom
        y *= (zoom * aspect_ratio)
    else:
        x *= zoom
        y *= zoom
        
    screen_x = (x + 1.0) * width / 2.0
    screen_y = (1.0 - y) * height / 2.0 
    return int(screen_x), int(screen_y)

def draw_ui(surface, sorted_edges, current_idx, edges_in_mst, total_vertices, slider_val, slider_rect, state_data, 
            nodes, renderer, ui_state):
    surface.fill((0,0,0,0))
    w, h = surface.get_size()
    interactive_rects = {}
    
    # --- 1. LABELS (Conditional) ---
    if total_vertices < 500:
        font = pygame.font.SysFont("Arial", 12, bold=True)
        weight_font = pygame.font.SysFont("Arial", 11)
        
        # Draw Weights (Only if show_weights is TRUE AND edge is not rejected)
        if ui_state['show_weights']:
            # We iterate by index because we need to check state_data[i]
            for i, (u, v, weight) in enumerate(sorted_edges):
                # If i >= current_idx, status is 0 (Unseen). Safe to draw.
                # If i < current_idx, check status.
                
                should_draw = True
                if i < current_idx:
                    status = state_data[i*2]
                    if status > 0.9 and status < 1.9: # Status 1.0 = Rejected
                        should_draw = False # HIDE LABEL
                
                if should_draw and int(u) < len(nodes) and int(v) < len(nodes):
                    p1 = nodes[int(u)]
                    p2 = nodes[int(v)]
                    s1 = project_point(p1, w, h, renderer.offset, renderer.zoom, renderer.aspect_ratio, renderer.correct_aspect)
                    s2 = project_point(p2, w, h, renderer.offset, renderer.zoom, renderer.aspect_ratio, renderer.correct_aspect)
                    
                    mid_x, mid_y = (s1[0] + s2[0]) // 2, (s1[1] + s2[1]) // 2
                    if 0 <= mid_x <= w and 0 <= mid_y <= h:
                        txt = weight_font.render(f"{int(weight)}", True, (0, 255, 255))
                        bg = pygame.Rect(mid_x, mid_y, txt.get_width(), txt.get_height())
                        pygame.draw.rect(surface, (0,0,0, 150), bg)
                        surface.blit(txt, (mid_x, mid_y))

        # Draw Node IDs
        if ui_state['show_ids']:
            for i, node in enumerate(nodes):
                sx, sy = project_point(node, w, h, renderer.offset, renderer.zoom, renderer.aspect_ratio, renderer.correct_aspect)
                if -20 <= sx <= w + 20 and -20 <= sy <= h + 20:
                    txt = font.render(str(i), True, (255, 255, 255))
                    rect = txt.get_rect(center=(sx, sy))
                    surface.blit(txt, rect)
    
    # --- 2. SLIDER ---
    pygame.draw.rect(surface, (40, 40, 40), slider_rect)
    pygame.draw.rect(surface, (200, 200, 200), slider_rect, 2)
    fill_width = int(slider_rect.width * slider_val)
    pygame.draw.rect(surface, (0, 200, 100), (slider_rect.x, slider_rect.y, fill_width, slider_rect.height))
    
    font = pygame.font.SysFont("Arial", 14)
    speed_text = "Step-by-Step" if slider_val < 0.05 else f"Auto Speed: {int(slider_val*100)}%"
    surface.blit(font.render(speed_text, True, (255,255,255)), (slider_rect.x, slider_rect.y - 20))

    # --- 3. STATS ---
    status_font = pygame.font.SysFont("Consolas", 18)
    target = total_vertices - 1
    status_txt = f"MST Edges: {edges_in_mst} / {target}"
    col = (0, 255, 0) if edges_in_mst >= target else (255, 255, 0)
    if current_idx >= len(sorted_edges) and edges_in_mst < target: 
        status_txt += " [FAILED]"
        col = (255, 0, 0)
    surface.blit(status_font.render(status_txt, True, col), (20, 20))
    
    # --- 4. TOGGLES (Checkboxes) ---
    # IDs Toggle
    cb_ids = pygame.Rect(20, 50, 20, 20)
    interactive_rects['toggle_ids'] = cb_ids
    pygame.draw.rect(surface, (50, 50, 50), cb_ids)
    pygame.draw.rect(surface, (200, 200, 200), cb_ids, 2)
    if ui_state['show_ids']:
        pygame.draw.lines(surface, (0, 255, 0), False, [(cb_ids.x+4, cb_ids.y+10), (cb_ids.x+8, cb_ids.y+16), (cb_ids.x+16, cb_ids.y+4)], 2)
    surface.blit(font.render("Show IDs", True, (255,255,255)), (cb_ids.right+10, cb_ids.y))
    
    # Weights Toggle
    cb_w = pygame.Rect(20, 80, 20, 20)
    interactive_rects['toggle_weights'] = cb_w
    pygame.draw.rect(surface, (50, 50, 50), cb_w)
    pygame.draw.rect(surface, (200, 200, 200), cb_w, 2)
    if ui_state['show_weights']:
        pygame.draw.lines(surface, (0, 255, 0), False, [(cb_w.x+4, cb_w.y+10), (cb_w.x+8, cb_w.y+16), (cb_w.x+16, cb_w.y+4)], 2)
    surface.blit(font.render("Show Weights", True, (255,255,255)), (cb_w.right+10, cb_w.y))

    # --- 5. EDGE QUEUE TABLE ---
    table_width = 240
    table_x = w - table_width if ui_state['show_table'] else w
    
    toggle_rect = pygame.Rect(table_x - 30, 10, 30, 30)
    interactive_rects['table_toggle'] = toggle_rect
    
    pygame.draw.rect(surface, (40, 40, 40), toggle_rect, border_top_left_radius=5, border_bottom_left_radius=5)
    arrow_txt = ">" if ui_state['show_table'] else "<"
    arrow_surf = font.render(arrow_txt, True, (255, 255, 255))
    surface.blit(arrow_surf, (toggle_rect.centerx - arrow_surf.get_width()//2, toggle_rect.centery - arrow_surf.get_height()//2))

    if ui_state['show_table'] and len(sorted_edges) > 0:
        pygame.draw.rect(surface, (0, 0, 0, 200), (table_x, 0, table_width, h))
        header = status_font.render("Edge Queue", True, (0, 255, 255))
        surface.blit(header, (table_x + 10, 10))
        
        row_height = 25
        start_y = 50
        max_rows = (h - start_y) // row_height
        start_idx = ui_state['scroll']
        
        for i in range(max_rows):
            idx = start_idx + i
            if idx >= len(sorted_edges): break
            u, v, w = sorted_edges[idx]
            
            bg_color = None
            text_color = (180, 180, 180)
            prefix = "   "
            
            if idx < current_idx:
                status = state_data[idx*2]
                if status > 1.9:
                    bg_color = (255, 215, 0, 80)
                    text_color = (255, 255, 200)
                    prefix = "MST "
                elif status > 0.9:
                    bg_color = (255, 0, 0, 150)
                    text_color = (255, 200, 200)
                    prefix = "DEL "
            elif idx == current_idx:
                bg_color = (0, 255, 100, 50)
                text_color = (255, 255, 255)
                prefix = "-> "
            
            row_rect = pygame.Rect(table_x, start_y + i * row_height, table_width, row_height)
            if bg_color:
                pygame.draw.rect(surface, bg_color, row_rect)
            
            row_txt = f"{prefix}{int(u)}-{int(v)} : {w:.1f}"
            surface.blit(font.render(row_txt, True, text_color), (table_x + 10, start_y + i * row_height + 5))
            
    return interactive_rects

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
    nodes = []
    
    current_edge_idx = 0
    mst_edges_count = 0
    speed_value = 0.0
    slider_rect = pygame.Rect(20, SCREEN_SIZE[1] - 50, 200, 30)
    ui_surface = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
    
    ui_state = {
        'scroll': 0,
        'show_table': True,
        'show_ids': True,     # DEFAULT ON
        'show_weights': True  # DEFAULT ON
    }
    ui_rects = {}
    
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
                ctx.viewport = (0, 0, event.w, event.h)
                if graph_renderer:
                    graph_renderer.aspect_ratio = event.w / event.h
                ui_surface = pygame.Surface((event.w, event.h), pygame.SRCALPHA)
                slider_rect.y = event.h - 50

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    current_state = STATE_EDIT
                    graph_renderer = None; circle_renderer = None
                    pygame.display.set_caption("Edit Mode")

                elif event.key == pygame.K_t and current_state == STATE_EDIT:
                    # LIMIT TEST
                    sorted_edges, sorted_geom, raw_nodes = logic.prepare_data(TEST_VERTICES, TEST_EDGES)
                    nodes = raw_nodes
                    n_vertices = len(nodes)
                    parent = np.arange(n_vertices)
                    rank = np.zeros(n_vertices, dtype=np.int32)
                    
                    graph_renderer = gui.GraphRenderer(ctx, len(sorted_edges), sorted_geom, correct_aspect=True)
                    w, h = screen.get_size()
                    graph_renderer.aspect_ratio = w / h
                    
                    circle_renderer = gui.CircleRenderer(ctx, nodes)
                    circle_renderer.update_state(np.arange(n_vertices), np.zeros(n_vertices))
                    
                    current_edge_idx = 0; mst_edges_count = 0
                    current_state = STATE_RUN; speed_value = TEST_SPEED
                    ui_state['show_ids'] = False # Disable for massive graph
                    ui_state['show_weights'] = False

                elif event.key == pygame.K_RETURN and current_state == STATE_EDIT:
                    # NORMAL RUN
                    raw_edges, geom, raw_nodes = graph_editor.export_data()
                    nodes = raw_nodes
                    
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
                        
                        graph_renderer = gui.GraphRenderer(ctx, len(sorted_edges), sorted_geom, correct_aspect=False)
                        w, h = screen.get_size()
                        graph_renderer.aspect_ratio = w / h

                        circle_renderer = gui.CircleRenderer(ctx, nodes)
                        circle_renderer.update_state(np.arange(n_vertices), np.zeros(n_vertices))
                        
                        current_edge_idx = 0; mst_edges_count = 0
                        current_state = STATE_RUN; speed_value = 0.0
                        
                        # Match UI state from Editor
                        ui_state['show_ids'] = graph_editor.show_ids
                        ui_state['show_weights'] = graph_editor.show_weights
                    else:
                        print("Empty Graph")

                elif event.key == pygame.K_LEFT and current_state == STATE_RUN:
                    if current_edge_idx > 0:
                        current_edge_idx -= 1
                        graph_renderer.update_states(current_edge_idx, 1, np.array([0.0, 0.0], dtype='f4'))
                        graph_renderer.state_data[current_edge_idx*2] = 0.0
                        graph_renderer.state_data[current_edge_idx*2+1] = 0.0
                        parent = np.arange(len(parent))
                        rank[:] = 0
                        mst_edges_count = logic.fast_forward_dsu(sorted_edges, current_edge_idx, parent, rank)
                        should_force_update = True
                        is_paused = True

                elif event.key == pygame.K_RIGHT and current_state == STATE_RUN:
                    is_paused = False 

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if current_state == STATE_RUN:
                    if 'table_toggle' in ui_rects and ui_rects['table_toggle'].collidepoint(event.pos):
                        ui_state['show_table'] = not ui_state['show_table']
                    elif 'toggle_ids' in ui_rects and ui_rects['toggle_ids'].collidepoint(event.pos):
                        ui_state['show_ids'] = not ui_state['show_ids']
                    elif 'toggle_weights' in ui_rects and ui_rects['toggle_weights'].collidepoint(event.pos):
                        ui_state['show_weights'] = not ui_state['show_weights']

            if event.type == pygame.MOUSEWHEEL and current_state == STATE_RUN:
                table_w = 240
                if ui_state['show_table'] and mouse_pos[0] > screen.get_size()[0] - table_w:
                    ui_state['scroll'] -= event.y 
                    ui_state['scroll'] = max(0, min(ui_state['scroll'], len(sorted_edges) - 10))
                else:
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
            should_force_update = False
            
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
                
                if ui_state['show_table']:
                     rows_visible = (screen.get_size()[1] - 50) // 25
                     if current_edge_idx > ui_state['scroll'] + rows_visible - 2:
                         ui_state['scroll'] = current_edge_idx - rows_visible + 2

            if frame_count == 1 or frame_count % 10 == 0 or should_step or 'should_force_update' in locals():
                 roots = logic.get_all_roots(parent)
                 statuses = logic.get_node_statuses(parent, rank)
                 valid_roots = roots[:len(circle_renderer.colors)]
                 valid_stats = statuses[:len(circle_renderer.colors)]
                 circle_renderer.update_state(valid_roots, valid_stats)

            graph_renderer.render()
            pt_size = 5.0 if len(parent) > 2000 else 12.0
            circle_renderer.render(graph_renderer.current_matrix, point_size=pt_size)
            
            ui_rects = draw_ui(ui_surface, sorted_edges, current_edge_idx, mst_edges_count, 
                    len(parent), speed_value, slider_rect, 
                    graph_renderer.state_data, nodes, graph_renderer, ui_state)
            
            ui_renderer.update_texture(ui_surface)
            ui_renderer.render()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()