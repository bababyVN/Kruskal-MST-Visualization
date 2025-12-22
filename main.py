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

TEST_VERTICES = 100_000 
TEST_EDGES = 1_000_000

def project_point(point, width, height, offset, zoom):
    x, y = point[0], point[1]
    screen_x = x * zoom + offset[0]
    screen_y = y * zoom + offset[1]
    return int(screen_x), int(screen_y)

def draw_ui(surface, sorted_edges, current_idx, edges_in_mst, total_vertices, mst_total_weight, slider_val, slider_rect, state_data, 
            nodes, labels, renderer, ui_state):
    surface.fill((0,0,0,0))
    w, h = surface.get_size()
    interactive_rects = {}
    
    # --- 1. LABELS ---
    if total_vertices < 500:
        font = pygame.font.SysFont("Arial", 12, bold=True)
        weight_font = pygame.font.SysFont("Arial", 11)
        
        if ui_state['show_weights']:
            for i, (u, v, weight) in enumerate(sorted_edges):
                should_draw = True
                if i < current_idx:
                    status = state_data[i*2]
                    if status > 0.9 and status < 1.9 and not ui_state['show_deleted']: should_draw = False
                else:
                    if not ui_state['show_unseen']: should_draw = False
                
                if should_draw and int(u) < len(nodes) and int(v) < len(nodes):
                    p1, p2 = nodes[int(u)], nodes[int(v)]
                    s1 = project_point(p1, w, h, renderer.offset, renderer.zoom)
                    s2 = project_point(p2, w, h, renderer.offset, renderer.zoom)
                    
                    mid_x, mid_y = (s1[0] + s2[0]) // 2, (s1[1] + s2[1]) // 2
                    if 0 <= mid_x <= w and 0 <= mid_y <= h:
                        txt = weight_font.render(f"{int(weight)}", True, (0, 255, 255))
                        bg = pygame.Rect(mid_x, mid_y, txt.get_width(), txt.get_height())
                        pygame.draw.rect(surface, (0,0,0, 150), bg)
                        surface.blit(txt, (mid_x, mid_y))

        if ui_state['show_ids']:
            for i, node in enumerate(nodes):
                sx, sy = project_point(node, w, h, renderer.offset, renderer.zoom)
                if -20 <= sx <= w + 20 and -20 <= sy <= h + 20:
                    lbl = labels[i] if i < len(labels) else str(i)
                    txt = font.render(lbl, True, (255, 255, 255))
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
    
    # RESET BUTTON
    btn_reset = pygame.Rect(slider_rect.right + 20, slider_rect.y, 80, 30)
    interactive_rects['reset'] = btn_reset
    pygame.draw.rect(surface, (200, 50, 50), btn_reset, border_radius=5)
    rst_txt = font.render("STOP", True, (255,255,255))
    surface.blit(rst_txt, rst_txt.get_rect(center=btn_reset.center))

    # --- 3. STATS (Top Left) ---
    status_font = pygame.font.SysFont("Consolas", 18)
    target = total_vertices - 1
    
    # Line 1: Edges Count
    status_txt = f"MST Edges: {edges_in_mst} / {target}"
    col = (0, 255, 0) if edges_in_mst >= target else (255, 255, 0)
    if current_idx >= len(sorted_edges) and edges_in_mst < target: 
        status_txt += " [FAILED]"
        col = (255, 0, 0)
    surface.blit(status_font.render(status_txt, True, col), (20, 15))
    
    # Line 2: Total Weight
    weight_txt = f"Total Weight: {mst_total_weight:.1f}"
    surface.blit(status_font.render(weight_txt, True, (0, 255, 255)), (20, 35))
    
    # --- TOGGLES ---
    def draw_checkbox(rect, label, state, key):
        pygame.draw.rect(surface, (50, 50, 50), rect)
        pygame.draw.rect(surface, (200, 200, 200), rect, 2)
        if state: pygame.draw.lines(surface, (0, 255, 0), False, [(rect.x+4, rect.y+10), (rect.x+8, rect.y+16), (rect.x+16, rect.y+4)], 2)
        surface.blit(font.render(label, True, (255,255,255)), (rect.right+10, rect.y))
        interactive_rects[key] = rect

    # Shifted down slightly to make room for weight stats
    draw_checkbox(pygame.Rect(20, 65, 20, 20), "Show IDs", ui_state['show_ids'], 'toggle_ids')
    draw_checkbox(pygame.Rect(20, 95, 20, 20), "Show Weights", ui_state['show_weights'], 'toggle_weights')
    draw_checkbox(pygame.Rect(160, 65, 20, 20), "Show Rejected (Red)", ui_state['show_deleted'], 'toggle_deleted')
    draw_checkbox(pygame.Rect(160, 95, 20, 20), "Show Pending (Grey)", ui_state['show_unseen'], 'toggle_unseen')

    # --- 4. EDGE QUEUE ---
    table_width = 240
    table_x = w - table_width if ui_state['show_table'] else w
    
    toggle_rect = pygame.Rect(table_x - 30, 10, 30, 30)
    interactive_rects['table_toggle'] = toggle_rect
    pygame.draw.rect(surface, (40, 40, 40), toggle_rect, border_top_left_radius=5, border_bottom_left_radius=5)
    arrow_txt = ">" if ui_state['show_table'] else "<"
    surface.blit(font.render(arrow_txt, True, (255,255,255)), (toggle_rect.centerx-5, toggle_rect.centery-8))

    if ui_state['show_table'] and len(sorted_edges) > 0:
        pygame.draw.rect(surface, (0, 0, 0, 200), (table_x, 0, table_width, h))
        surface.blit(status_font.render("Edge Queue", True, (0, 255, 255)), (table_x + 10, 10))
        
        row_height = 25
        start_y = 50
        max_rows = (h - start_y) // row_height
        
        if ui_state['scroll'] > len(sorted_edges) - 1:
            ui_state['scroll'] = 0
        start_idx = ui_state['scroll']
        
        for i in range(max_rows):
            idx = start_idx + i
            if idx >= len(sorted_edges): break
            u, v, w = sorted_edges[idx]
            
            lbl_u = labels[int(u)] if int(u) < len(labels) else str(int(u))
            lbl_v = labels[int(v)] if int(v) < len(labels) else str(int(v))
            
            bg_color = None
            text_color = (180, 180, 180)
            prefix = "   "
            if idx < current_idx:
                status = state_data[idx*2]
                if status > 1.9:
                    bg_color = (255, 215, 0, 80)
                    text_color = (255, 255, 200); prefix = "MST "
                elif status > 0.9:
                    bg_color = (255, 0, 0, 150)
                    text_color = (255, 200, 200); prefix = "DEL "
            elif idx == current_idx:
                bg_color = (0, 255, 100, 50)
                text_color = (255, 255, 255); prefix = "-> "
            
            if bg_color: pygame.draw.rect(surface, bg_color, (table_x, start_y + i * row_height, table_width, row_height))
            surface.blit(font.render(f"{prefix}{lbl_u}-{lbl_v} : {w:.1f}", True, text_color), (table_x + 10, start_y + i * row_height + 5))
            
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
    labels = []
    
    current_edge_idx = 0
    mst_edges_count = 0
    mst_total_weight = 0.0 # NEW: Total weight tracker
    
    speed_value = 0.0
    slider_rect = pygame.Rect(20, SCREEN_SIZE[1] - 50, 200, 30)
    ui_surface = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
    
    ui_state = {
        'scroll': 0, 'show_table': True, 
        'show_ids': True, 'show_weights': True,
        'show_deleted': True, 
        'show_unseen': True
    }
    ui_rects = {}
    clock = pygame.time.Clock()
    is_paused = True
    frame_count = 0
    
    run_zoom = 1.0
    run_offset = np.array([0.0, 0.0], dtype='f4')
    is_panning = False
    pan_start = np.array([0,0], dtype='f4')

    while True:
        events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.VIDEORESIZE:
                ctx.viewport = (0, 0, event.w, event.h)
                if graph_renderer: graph_renderer.set_camera(run_zoom, run_offset, event.w, event.h)
                ui_surface = pygame.Surface((event.w, event.h), pygame.SRCALPHA)
                slider_rect.y = event.h - 50

            # --- TRANSITIONS ---
            should_start_run = False
            is_limit_test = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and current_state == STATE_EDIT: should_start_run = True
                elif event.key == pygame.K_t and current_state == STATE_EDIT: 
                    should_start_run = True; is_limit_test = True
            
            if current_state == STATE_EDIT and graph_editor.trigger_run:
                should_start_run = True; graph_editor.trigger_run = False

            if should_start_run:
                run_zoom = graph_editor.zoom
                run_offset = graph_editor.offset.copy()
                
                # RESET STATE
                ui_state['scroll'] = 0 
                mst_total_weight = 0.0 
                
                if is_limit_test:
                    run_zoom = 0.5; run_offset = np.array([screen.get_width()/2, screen.get_height()/2], dtype='f4')
                    sorted_edges, sorted_geom, raw_nodes = logic.prepare_data(TEST_VERTICES, TEST_EDGES)
                    nodes = raw_nodes
                    labels = [str(i) for i in range(len(nodes))]
                    
                    n_vertices = len(nodes)
                    parent = np.arange(n_vertices)
                    rank = np.zeros(n_vertices, dtype=np.int32)
                    graph_renderer = gui.GraphRenderer(ctx, len(sorted_edges), sorted_geom)
                    circle_renderer = gui.CircleRenderer(ctx, nodes)
                    circle_renderer.update_state(np.arange(n_vertices), np.zeros(n_vertices))
                    speed_value = 0.15
                    ui_state['show_ids'] = False
                    ui_state['show_weights'] = False
                else:
                    raw_edges, geom, raw_nodes, raw_labels = graph_editor.export_data()
                    if raw_edges is not None and len(raw_edges) > 0:
                        nodes = raw_nodes
                        labels = raw_labels
                        
                        sorted_indices = np.argsort(raw_edges[:, 2])
                        sorted_edges = raw_edges[sorted_indices]
                        sorted_geom = np.empty_like(geom)
                        for new_i, old_i in enumerate(sorted_indices):
                            sorted_geom[new_i*2] = geom[old_i*2]
                            sorted_geom[new_i*2+1] = geom[old_i*2+1]
                        
                        n_vertices = len(nodes)
                        parent = np.arange(n_vertices)
                        rank = np.zeros(n_vertices, dtype=np.int32)
                        
                        graph_renderer = gui.GraphRenderer(ctx, len(sorted_edges), sorted_geom)
                        circle_renderer = gui.CircleRenderer(ctx, nodes)
                        circle_renderer.update_state(np.arange(n_vertices), np.zeros(n_vertices))
                        
                        speed_value = 0.0
                        ui_state['show_ids'] = graph_editor.show_ids
                        ui_state['show_weights'] = graph_editor.show_weights
                    else:
                        print("Empty Graph"); continue

                w, h = screen.get_size()
                graph_renderer.set_camera(run_zoom, run_offset, w, h)
                current_edge_idx = 0; mst_edges_count = 0
                current_state = STATE_RUN

            # --- RUN INPUTS ---
            if current_state == STATE_RUN:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_ui = False
                    if slider_rect.collidepoint(event.pos): clicked_ui = True
                    for r in ui_rects.values():
                        if r.collidepoint(event.pos): clicked_ui = True
                    
                    if event.button == 1:
                        if clicked_ui:
                            if 'reset' in ui_rects and ui_rects['reset'].collidepoint(event.pos): current_state = STATE_EDIT
                            elif 'table_toggle' in ui_rects and ui_rects['table_toggle'].collidepoint(event.pos): ui_state['show_table'] = not ui_state['show_table']
                            elif 'toggle_ids' in ui_rects and ui_rects['toggle_ids'].collidepoint(event.pos): ui_state['show_ids'] = not ui_state['show_ids']
                            elif 'toggle_weights' in ui_rects and ui_rects['toggle_weights'].collidepoint(event.pos): ui_state['show_weights'] = not ui_state['show_weights']
                            elif 'toggle_deleted' in ui_rects and ui_rects['toggle_deleted'].collidepoint(event.pos): ui_state['show_deleted'] = not ui_state['show_deleted']
                            elif 'toggle_unseen' in ui_rects and ui_rects['toggle_unseen'].collidepoint(event.pos): ui_state['show_unseen'] = not ui_state['show_unseen']
                        else:
                            is_panning = True
                            pan_start = np.array(mouse_pos, dtype='f4')
                    elif event.button == 2:
                        is_panning = True
                        pan_start = np.array(mouse_pos, dtype='f4')

                elif event.type == pygame.MOUSEBUTTONUP: 
                    is_panning = False

                elif event.type == pygame.MOUSEMOTION and is_panning:
                    delta = np.array(mouse_pos, dtype='f4') - pan_start
                    run_offset += delta
                    pan_start = np.array(mouse_pos, dtype='f4')
                    graph_renderer.set_camera(run_zoom, run_offset, screen.get_width(), screen.get_height())

                elif event.type == pygame.MOUSEWHEEL:
                    if ui_state['show_table'] and mouse_pos[0] > screen.get_size()[0] - 240:
                        ui_state['scroll'] -= event.y 
                        ui_state['scroll'] = max(0, min(ui_state['scroll'], len(sorted_edges) - 10))
                    else:
                        zoom_factor = 1.1 if event.y > 0 else 0.9
                        m_arr = np.array(mouse_pos, dtype='f4')
                        world_before = (m_arr - run_offset) / run_zoom
                        run_zoom = max(0.0001, run_zoom * zoom_factor)
                        run_offset = m_arr - (world_before * run_zoom)
                        graph_renderer.set_camera(run_zoom, run_offset, screen.get_width(), screen.get_height())

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: current_state = STATE_EDIT
                    elif event.key == pygame.K_LEFT:
                        if current_edge_idx > 0:
                            current_edge_idx -= 1
                            graph_renderer.update_states(current_edge_idx, 1, np.array([0.0, 0.0], dtype='f4'))
                            graph_renderer.state_data[current_edge_idx*2] = 0.0
                            graph_renderer.state_data[current_edge_idx*2+1] = 0.0
                            parent = np.arange(len(parent)); rank[:] = 0
                            
                            # UNDO LOGIC: Recalculate weight
                            mst_edges_count, mst_total_weight = logic.fast_forward_dsu(sorted_edges, current_edge_idx, parent, rank)
                            
                            should_force_update = True; is_paused = True
                    elif event.key == pygame.K_RIGHT: is_paused = False

            if current_state == STATE_EDIT:
                graph_editor.handle_event(event)

        if current_state == STATE_RUN and mouse_buttons[0] and not is_panning:
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
            
            if mst_edges_count < (len(parent)-1) and current_edge_idx < len(sorted_edges):
                if speed_value < 0.05:
                    if not is_paused: should_step = True; is_paused = True
                else:
                    batch_size = int(1 + (speed_value * 50)**3); should_step = True

            if should_step:
                # FORWARD LOGIC: Add Weight
                processed, added, w_added = logic.process_batch(sorted_edges, current_edge_idx, batch_size, parent, rank, graph_renderer.state_data)
                start_byte = current_edge_idx * 2 * 4
                data_slice = graph_renderer.state_data[current_edge_idx*2 : (current_edge_idx+processed)*2]
                graph_renderer.state_vbo.write(data_slice.tobytes(), offset=start_byte)
                current_edge_idx += processed
                mst_edges_count += added
                mst_total_weight += w_added # Add new weight
                
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

            graph_renderer.set_visibility(ui_state['show_deleted'], ui_state['show_unseen'])
            
            graph_renderer.render()
            pt_size = 5.0 if len(parent) > 2000 else 12.0
            circle_renderer.render(graph_renderer.current_matrix, point_size=pt_size, zoom=graph_renderer.zoom)
            
            # Pass new mst_total_weight to UI
            ui_rects = draw_ui(ui_surface, sorted_edges, current_edge_idx, mst_edges_count, 
                    len(parent), mst_total_weight, speed_value, slider_rect, 
                    graph_renderer.state_data, nodes, labels, graph_renderer, ui_state)
            
            ui_renderer.update_texture(ui_surface)
            ui_renderer.render()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()