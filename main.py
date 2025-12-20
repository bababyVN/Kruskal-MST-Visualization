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

def draw_slider(surface, value, rect):
    """Draws a simple slider on a Pygame surface."""
    # Background
    pygame.draw.rect(surface, (50, 50, 50), rect)
    pygame.draw.rect(surface, (200, 200, 200), rect, 2)
    
    # Fill
    fill_width = int(rect.width * value)
    fill_rect = pygame.Rect(rect.x, rect.y, fill_width, rect.height)
    pygame.draw.rect(surface, (0, 200, 100), fill_rect)
    
    # Label
    font = pygame.font.SysFont("Arial", 14)
    text = font.render(f"Speed: {int(value * 100)}%", True, (255, 255, 255))
    surface.blit(text, (rect.x + 5, rect.y + 5))

def main():
    # 1. Init OpenGL Window
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    screen = pygame.display.set_mode(SCREEN_SIZE, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("Kruskal Sandbox")

    ctx = moderngl.create_context()

    # 2. Components
    graph_editor = editor.GraphEditor(*SCREEN_SIZE)
    ui_renderer = gui.TextureRenderer(ctx) 
    
    # Simulation components (created on Run)
    graph_renderer = None
    circle_renderer = None
    
    # Logic State
    current_state = STATE_EDIT
    sorted_edges = []
    parent = []
    rank = []
    current_edge_idx = 0
    paused = False
    
    # UI State
    speed_value = 0.05 # 0.0 to 1.0
    slider_rect = pygame.Rect(20, SCREEN_SIZE[1] - 50, 200, 30)
    ui_surface = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
    
    clock = pygame.time.Clock()

    while True:
        # --- INPUT ---
        events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # RESET
                    current_state = STATE_EDIT
                    graph_renderer = None
                    circle_renderer = None
                    pygame.display.set_caption("Edit Mode - [L]oad | [Enter] Run | [Right Click] Edit Weight")

                elif event.key == pygame.K_RETURN and current_state == STATE_EDIT:
                    # SWITCH TO RUN
                    raw_edges, geom, nodes = graph_editor.export_data()
                    if raw_edges is not None and len(raw_edges) > 0:
                        # Logic Init
                        sorted_indices = np.argsort(raw_edges[:, 2])
                        sorted_edges = raw_edges[sorted_indices]
                        
                        sorted_geom = np.empty_like(geom)
                        for new_i, old_i in enumerate(sorted_indices):
                            sorted_geom[new_i*2] = geom[old_i*2]
                            sorted_geom[new_i*2+1] = geom[old_i*2+1]

                        n_vertices = int(np.max(sorted_edges[:, :2])) + 1
                        parent = np.arange(n_vertices + 1)
                        rank = np.zeros(n_vertices + 1, dtype=np.int32)
                        
                        # Renderers Init
                        graph_renderer = gui.GraphRenderer(ctx, len(sorted_edges), sorted_geom)
                        circle_renderer = gui.CircleRenderer(ctx, nodes)
                        
                        current_edge_idx = 0
                        current_state = STATE_RUN
                        pygame.display.set_caption("Simulation Mode - [Space] Pause | [R] Reset")

                elif event.key == pygame.K_l and current_state == STATE_EDIT:
                    graph_editor.load_from_file()

                elif event.key == pygame.K_SPACE and current_state == STATE_RUN:
                    paused = not paused
                    
            if event.type == pygame.MOUSEWHEEL and current_state == STATE_RUN:
                graph_renderer.zoom *= 1.1 if event.y > 0 else 0.9

            # Editor Input
            if current_state == STATE_EDIT:
                graph_editor.handle_event(event)

        # Slider Input (Always check if in Run mode)
        if current_state == STATE_RUN:
            if mouse_buttons[0]: # Left click held
                if slider_rect.collidepoint(mouse_pos):
                    # Update slider value
                    rel_x = mouse_pos[0] - slider_rect.x
                    speed_value = max(0.001, min(1.0, rel_x / slider_rect.width))

        # --- DRAWING ---
        ctx.clear(0.1, 0.1, 0.1)

        if current_state == STATE_EDIT:
            surface = graph_editor.draw()
            ui_renderer.update_texture(surface)
            ui_renderer.render()

        elif current_state == STATE_RUN:
            # 1. Calculate Speed
            # Min: 1 edge/frame, Max: 5000 edges/frame
            batch_size = int(1 + (speed_value * 100)**2) 
            
            # 2. Logic Step
            if not paused and current_edge_idx < len(sorted_edges):
                processed = logic.process_batch(
                    sorted_edges, current_edge_idx, batch_size, 
                    parent, rank, graph_renderer.state_data
                )
                graph_renderer.update_states(
                    current_edge_idx, processed, 
                    graph_renderer.state_data[current_edge_idx*2 : (current_edge_idx+processed)*2]
                )
                current_edge_idx += processed

            # 3. Render Graph & Nodes
            graph_renderer.render() # Updates camera matrix
            circle_renderer.render(graph_renderer.current_matrix) # Uses same matrix
            
            # 4. Render UI (Slider)
            ui_surface.fill((0,0,0,0))
            draw_slider(ui_surface, speed_value, slider_rect)
            
            # Draw Progress Text
            font = pygame.font.SysFont("Arial", 20)
            txt = font.render(f"Edges: {current_edge_idx} / {len(sorted_edges)}", True, (255, 255, 0))
            ui_surface.blit(txt, (20, 20))
            
            ui_renderer.update_texture(ui_surface)
            ui_renderer.render()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()