import pygame
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

class GraphEditor:
    def __init__(self, width, height):
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.width = width
        self.height = height
        
        self.nodes = [] 
        self.edges = [] 
        
        # Interaction
        self.selected_node = None
        self.hovered_node = None
        self.dragging_edge = False
        
        # Toggle States
        self.show_ids = True
        self.show_weights = True
        
        self.font = pygame.font.SysFont("Arial", 16)
        self.node_font = pygame.font.SysFont("Arial", 12, bold=True)
        
        # Checkboxes
        self.cb_ids_rect = pygame.Rect(10, 10, 20, 20)
        self.cb_w_rect = pygame.Rect(10, 40, 20, 20)
        
        root = tk.Tk()
        root.withdraw()

    def load_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text/CSV", "*.txt *.csv")])
        if not file_path: return

        try:
            new_edges = []
            max_node_idx = 0
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().replace(',', ' ').split()
                    if len(parts) >= 3:
                        u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                        new_edges.append([u, v, w])
                        max_node_idx = max(max_node_idx, u, v)
            
            self.nodes = []
            self.edges = new_edges
            n_nodes = max_node_idx + 1
            radius = min(self.width, self.height) * 0.4
            center_x, center_y = self.width // 2, self.height // 2
            
            for i in range(n_nodes):
                angle = 2 * math.pi * i / n_nodes
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                self.nodes.append([x, y])
            print(f"Loaded {len(self.edges)} edges.")
            
        except Exception as e:
            print(f"Error loading file: {e}")

    def handle_event(self, event):
        mouse_pos = pygame.mouse.get_pos()
        self.hovered_node = self._find_nearest_node(mouse_pos)
        hovered_edge_idx = self._find_nearest_edge_weight(mouse_pos)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: 
                # Check UI first
                if self.cb_ids_rect.collidepoint(mouse_pos):
                    self.show_ids = not self.show_ids
                    return
                if self.cb_w_rect.collidepoint(mouse_pos):
                    self.show_weights = not self.show_weights
                    return

                if self.hovered_node is not None:
                    self.selected_node = self.hovered_node
                    self.dragging_edge = True
                else:
                    self.nodes.append(list(mouse_pos))
                    
            elif event.button == 3: 
                if hovered_edge_idx is not None:
                    current_w = self.edges[hovered_edge_idx][2]
                    new_w = 1.0 if current_w >= 20 else (current_w + 5.0)
                    self.edges[hovered_edge_idx][2] = new_w
                elif self.hovered_node is not None:
                    self._delete_node(self.hovered_node)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.dragging_edge:
                if self.hovered_node is not None and self.hovered_node != self.selected_node:
                    if not self._edge_exists(self.selected_node, self.hovered_node):
                        self.edges.append([self.selected_node, self.hovered_node, 5.0])
                self.dragging_edge = False
                self.selected_node = None

    def _find_nearest_node(self, pos, threshold=15):
        for i, (nx, ny) in enumerate(self.nodes):
            if math.hypot(nx - pos[0], ny - pos[1]) < threshold: return i
        return None

    def _find_nearest_edge_weight(self, pos, threshold=20):
        if not self.show_weights: return None
        for i, (u, v, w) in enumerate(self.edges):
            if u >= len(self.nodes) or v >= len(self.nodes): continue
            start, end = self.nodes[u], self.nodes[v]
            mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            if math.hypot(mid_x - pos[0], mid_y - pos[1]) < threshold: return i
        return None

    def _edge_exists(self, u, v):
        for edge in self.edges:
            if (edge[0] == u and edge[1] == v) or (edge[0] == v and edge[1] == u): return True
        return False

    def _delete_node(self, node_idx):
        self.nodes.pop(node_idx)
        self.edges = [e for e in self.edges if e[0] != node_idx and e[1] != node_idx]
        for edge in self.edges:
            if edge[0] > node_idx: edge[0] -= 1
            if edge[1] > node_idx: edge[1] -= 1

    def draw(self):
        self.surface.fill((0, 0, 0, 0)) 
        
        # --- UI Checkboxes ---
        # IDs
        pygame.draw.rect(self.surface, (50, 50, 50), self.cb_ids_rect)
        pygame.draw.rect(self.surface, (200, 200, 200), self.cb_ids_rect, 2)
        if self.show_ids:
             pygame.draw.lines(self.surface, (0, 255, 0), False, 
                [(self.cb_ids_rect.x+4, self.cb_ids_rect.y+10), 
                 (self.cb_ids_rect.x+8, self.cb_ids_rect.y+16), 
                 (self.cb_ids_rect.x+16, self.cb_ids_rect.y+4)], 2)
        lbl_id = self.font.render("Show IDs", True, (255, 255, 255))
        self.surface.blit(lbl_id, (self.cb_ids_rect.right + 10, self.cb_ids_rect.y))
        
        # Weights
        pygame.draw.rect(self.surface, (50, 50, 50), self.cb_w_rect)
        pygame.draw.rect(self.surface, (200, 200, 200), self.cb_w_rect, 2)
        if self.show_weights:
             pygame.draw.lines(self.surface, (0, 255, 0), False, 
                [(self.cb_w_rect.x+4, self.cb_w_rect.y+10), 
                 (self.cb_w_rect.x+8, self.cb_w_rect.y+16), 
                 (self.cb_w_rect.x+16, self.cb_w_rect.y+4)], 2)
        lbl_w = self.font.render("Show Weights", True, (255, 255, 255))
        self.surface.blit(lbl_w, (self.cb_w_rect.right + 10, self.cb_w_rect.y))

        # --- Graph ---
        for u, v, w in self.edges:
            if u < len(self.nodes) and v < len(self.nodes):
                start, end = self.nodes[u], self.nodes[v]
                pygame.draw.line(self.surface, (200, 200, 200), start, end, 2)
                
                if self.show_weights:
                    mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                    label = f"{int(w)}"
                    text = self.font.render(label, True, (0, 255, 255), (30, 30, 30))
                    rect = text.get_rect(center=(mid_x, mid_y))
                    self.surface.blit(text, rect)

        if self.dragging_edge and self.selected_node is not None:
            pygame.draw.line(self.surface, (100, 255, 100), self.nodes[self.selected_node], pygame.mouse.get_pos(), 1)

        for i, (x, y) in enumerate(self.nodes):
            color = (255, 100, 100) if i == self.hovered_node else (50, 150, 255)
            pygame.draw.circle(self.surface, color, (int(x), int(y)), 12)
            
            if self.show_ids:
                id_text = self.node_font.render(str(i), True, (255, 255, 255))
                id_rect = id_text.get_rect(center=(int(x), int(y)))
                self.surface.blit(id_text, id_rect)
            
        return self.surface

    def export_data(self):
        if not self.nodes: return None, None, None
        pixel_coords = np.array(self.nodes, dtype='f4')
        norm_coords = np.zeros_like(pixel_coords)
        norm_coords[:, 0] = (pixel_coords[:, 0] / self.width) * 2 - 1
        norm_coords[:, 1] = -((pixel_coords[:, 1] / self.height) * 2 - 1)
        edges_array = np.array(self.edges, dtype=np.float64)
        line_geometry = np.empty((len(self.edges) * 2, 2), dtype='f4')
        for i, (u, v, w) in enumerate(self.edges):
            line_geometry[i*2] = norm_coords[u]
            line_geometry[i*2+1] = norm_coords[v]
        return edges_array, line_geometry, norm_coords