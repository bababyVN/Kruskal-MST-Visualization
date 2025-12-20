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

TEST_VERTICES = 5000 
TEST_EDGES = 25000

def project_point(point, width, height, offset, zoom, aspect_ratio, correct_aspect):
    """
    Transforms a point from World Space (-1 to 1) to Screen Space (pixels).
    Must match the vertex shader logic in gui.py!
    """
    x, y = point[0], point[1]
    
    # 1. Translate
    x += offset[0]
    y += offset[1]
    
    # 2. Scale (Zoom & Aspect)
    if correct_aspect and aspect_ratio > 1.0:
        x *= (zoom / aspect_ratio)
        y *= zoom
    elif correct_aspect:
        x *= zoom
        y *= (zoom * aspect_ratio)
    else:
        # If Editor Mode, we didn't correct aspect, so scaling is uniform
        x *= zoom
        y *= zoom
        
    # 3. Viewport (Clip Space -1..1 to Screen Space 0..W)
    screen_x = (x + 1.0) * width / 2.0
    screen_y = (1.0 - y) * height / 2.0 # Flip Y
    
    return int(screen_x), int(screen_y)

def draw_ui(surface, sorted_edges, current_idx, edges_in_mst, total_vertices, slider_val, slider_rect, state_data, 
            nodes, renderer):
    """
    Draws UI overlay including labels for nodes/edges in the 3D view.
    """
    surface.fill((0,0,0,0))
    w, h = surface.get_size()
    
    # --- 1. LABELS (Nodes & Weights) ---
    # Only draw if graph is small enough to be readable/performant
    if total_vertices < 500:
        font = pygame.font.SysFont("Arial", 12, bold=True)
        weight_font = pygame.font.SysFont("Arial", 11)
        
        # Draw Edge Weights
        # (We iterate all edges; for performance on medium graphs, maybe limit this)
        for u, v, weight in sorted_edges:
            # Get positions of u and v
            if int(u) < len(nodes) and int(v) < len(nodes):
                p1 = nodes[int(u)]
                p2 = nodes[int(v)]
                
                # Project to screen
                s1 = project_point(p1, w, h, renderer.offset, renderer.zoom, renderer.aspect_ratio, renderer.correct_aspect)
                s2 = project_point(p2, w, h, renderer.offset, renderer.zoom, renderer.aspect_ratio, renderer.correct_aspect)
                
                # Midpoint
                mid_x = (s1[0] + s2[0]) // 2
                mid_y = (s1[1] + s2[1]) // 2
                
                # Simple check if on screen
                if 0 <= mid_x <= w and 0 <= mid_y <= h:
                    txt = weight_font.render(f"{int(weight)}", True, (0, 255, 255))
                    # Draw a small black box behind text for readability
                    bg = pygame.Rect(mid_x, mid_y, txt.get_width(), txt.get_height())
                    pygame.draw.rect(surface, (0,0,0, 150), bg)
                    surface.blit(txt, (mid_x, mid_y))

        # Draw Node Numbers
        for i, node in enumerate(nodes):
            sx, sy = project_point(node, w, h, renderer.offset, renderer.zoom, renderer.aspect_ratio, renderer.correct_aspect)
            
            # Culling: Only draw if on screen
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
    
    if edges_in_mst >= target:
        status_txt += " [COMPLETE]"
        col = (0, 255, 0)
    elif current_idx >= len(sorted_edges):
        status_txt += " [FAILED]"
        col = (255, 0, 0)
    else:
        col = (255, 255, 0)
        
    surface.blit(status_font.render(status_txt, True, col), (20, 20))

    # --- 4. EDGE QUEUE ---
    if len(sorted_edges) > 0:
        table_width = 240
        table_x = SCREEN_SIZE[0] - table_width
        pygame.draw.rect(surface, (0, 0, 0, 200), (table_x, 0, table_width, SCREEN_SIZE[1]))
        header = status_font.render("Edge Queue", True, (0, 255, 255))
        surface.blit(header, (table_x + 10, 10))
        
        row_height = 25
        start_y = 50
        start_display_idx = max(0, current_idx - 1)
        
        for i in range(25):
            idx = start_display_idx + i
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
            
            if bg_color:
                pygame.draw.rect(surface, bg_color, (table_x, start_y + i * row_height, table_width, row_height))
            
            row_txt = f"{prefix}{int(u)}-{int(v)} : {w:.1f}"
            surface.blit(font.render(row_txt, True, text_color), (table_x + 10, start_y + i * row_height + 5))

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
    nodes = [] # Store nodes here for UI drawing
    
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
                ctx.viewport = (0, 0, event.w, event.h)
                if graph_renderer:
                    graph_renderer.aspect_ratio = event.w / event.h

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    current_state = STATE_EDIT
                    graph_renderer = None; circle_renderer = None
                    pygame.display.set_caption("Edit Mode")

                elif event.key == pygame.K_t and current_state == STATE_EDIT:
                    # LIMIT TEST
                    sorted_edges, sorted_geom, raw_nodes = logic.prepare_data(TEST_VERTICES, TEST_EDGES)
                    nodes = raw_nodes # Save for UI
                    
                    n_vertices = len(nodes)
                    parent = np.arange(n_vertices)
                    rank = np.zeros(n_vertices, dtype=np.int32)
                    
                    graph_renderer = gui.GraphRenderer(ctx, len(sorted_edges), sorted_geom, correct_aspect=True)
                    w, h = screen.get_size()
                    graph_renderer.aspect_ratio = w / h
                    
                    circle_renderer = gui.CircleRenderer(ctx, nodes)
                    circle_renderer.update_state(np.arange(n_vertices), np.zeros(n_vertices))
                    
                    current_edge_idx = 0; mst_edges_count = 0
                    current_state = STATE_RUN; speed_value = 0.15

                elif event.key == pygame.K_RETURN and current_state == STATE_EDIT:
                    # NORMAL RUN
                    raw_edges, geom, raw_nodes = graph_editor.export_data() # raw_nodes are normalized -1..1
                    nodes = raw_nodes # Save for UI
                    
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

            if frame_count == 1 or frame_count % 10 == 0 or should_step:
                 roots = logic.get_all_roots(parent)
                 statuses = logic.get_node_statuses(parent, rank)
                 valid_roots = roots[:len(circle_renderer.colors)]
                 valid_stats = statuses[:len(circle_renderer.colors)]
                 circle_renderer.update_state(valid_roots, valid_stats)

            graph_renderer.render()
            pt_size = 5.0 if len(parent) > 2000 else 12.0
            circle_renderer.render(graph_renderer.current_matrix, point_size=pt_size)
            
            # --- Draw UI ---
            draw_ui(ui_surface, sorted_edges, current_edge_idx, mst_edges_count, 
                    len(parent), speed_value, slider_rect, 
                    graph_renderer.state_data,
                    nodes, graph_renderer) # Pass nodes and renderer for projection
            
            ui_renderer.update_texture(ui_surface)
            ui_renderer.render()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()