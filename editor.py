import pygame
import numpy as np
import tkinter as tk
from tkinter import filedialog
from config import *

class GraphEditor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Data
        self.nodes = [] # {'pos': [x,y], 'label': "A"}
        self.edges = [] # {'u': 0, 'v': 1, 'weight': 10.0}
        
        # Camera
        self.offset = np.array([width/2, height/2], dtype='f4')
        self.zoom = 1.0
        
        # State
        self.current_tool = TOOL_SELECT
        self.selection = None 
        self.hover_item = None
        self.dragging_node = None
        self.panning = False
        self.pan_start = (0,0)
        self.drag_edge_start = None
        self.mouse_world = (0,0)
        
        # UI State
        self.trigger_run = False
        self.editing_text = False 
        self.input_buffer = ""
        self.show_ids = True
        self.show_weights = True
        
        # Assets
        self.font = pygame.font.SysFont("Arial", 14)
        self.bold_font = pygame.font.SysFont("Arial", 14, bold=True)
        self.icon_font = pygame.font.SysFont("Segoe UI Symbol", 20) 
        
        # Toolbar Layout
        self.toolbar_h = 50
        self.tools = [
            {'id': TOOL_SELECT, 'icon': "➤", 'tip': "Move"},
            {'id': TOOL_POINT,  'icon': "●", 'tip': "Point"},
            {'id': TOOL_EDGE,   'icon': "╱", 'tip': "Segment"},
            {'id': TOOL_DELETE, 'icon': "✖", 'tip': "Delete"},
            {'id': TOOL_PAN,    'icon': "✋", 'tip': "Pan View"}
        ]
        
        self.btn_toggle_ids = pygame.Rect(width - 250, 12, 80, 26)
        self.btn_toggle_w = pygame.Rect(width - 160, 12, 80, 26)
        self.btn_run = pygame.Rect(width - 70, 10, 60, 30)
        
        self.root = tk.Tk()
        self.root.withdraw()

    # --- COORDINATES ---
    def screen_to_world(self, pos):
        return (np.array(pos, dtype='f4') - self.offset) / self.zoom

    def world_to_screen(self, pos):
        return (np.array(pos, dtype='f4') * self.zoom) + self.offset

    # --- INPUT ---
    def handle_event(self, event):
        m_pos = pygame.mouse.get_pos()
        self.mouse_world = self.screen_to_world(m_pos)
        self.hover_item = self._hit_test(self.mouse_world)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if m_pos[1] < self.toolbar_h:
                    self._handle_toolbar_click(m_pos); return
                
                # Check Inspector
                if self.selection:
                    insp_rect = pygame.Rect(20, self.height - 120, 200, 100)
                    if insp_rect.collidepoint(m_pos):
                        self.editing_text = True
                        if self.selection['type'] == 'node': self.input_buffer = self.nodes[self.selection['index']]['label']
                        else: self.input_buffer = str(int(self.edges[self.selection['index']]['weight']))
                        return
                
                self.editing_text = False 
                
                if self.current_tool == TOOL_SELECT:
                    if self.hover_item:
                        self.selection = self.hover_item
                        if self.hover_item['type'] == 'node': self.dragging_node = self.hover_item['index']
                    else: self.selection = None
                        
                elif self.current_tool == TOOL_POINT:
                    self.nodes.append({'pos': self.mouse_world, 'label': str(len(self.nodes))})
                    
                elif self.current_tool == TOOL_EDGE:
                    if self.hover_item and self.hover_item['type'] == 'node': self.drag_edge_start = self.hover_item['index']
                        
                elif self.current_tool == TOOL_DELETE:
                    if self.hover_item:
                        if self.hover_item['type'] == 'node': self._delete_node(self.hover_item['index'])
                        else: self._delete_edge(self.hover_item['index'])
                        
                elif self.current_tool == TOOL_PAN:
                    self.panning = True; self.pan_start = np.array(m_pos, dtype='f4')

            elif event.button == 2: 
                self.panning = True; self.pan_start = np.array(m_pos, dtype='f4')
            elif event.button == 3: 
                self.drag_edge_start = None; self.dragging_node = None

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self.current_tool == TOOL_EDGE and self.drag_edge_start is not None:
                    if self.hover_item and self.hover_item['type'] == 'node':
                        u, v = self.drag_edge_start, self.hover_item['index']
                        if u != v: self._create_edge(u, v)
                    self.drag_edge_start = None
                self.dragging_node = None; self.panning = False
            elif event.button == 2: self.panning = False

        elif event.type == pygame.MOUSEMOTION:
            if self.panning:
                self.offset += np.array(m_pos, dtype='f4') - self.pan_start
                self.pan_start = np.array(m_pos, dtype='f4')
            elif self.dragging_node is not None:
                self.nodes[self.dragging_node]['pos'] = self.mouse_world

        elif event.type == pygame.MOUSEWHEEL:
            zoom_factor = 1.1 if event.y > 0 else 0.9
            world_before = self.screen_to_world(m_pos)
            self.zoom = max(0.0001, self.zoom * zoom_factor)
            self.offset += (np.array(m_pos) - self.world_to_screen(world_before))

        elif event.type == pygame.KEYDOWN:
            if self.editing_text:
                if event.key == pygame.K_RETURN: self.editing_text = False
                elif event.key == pygame.K_BACKSPACE: self.input_buffer = self.input_buffer[:-1]
                else: self.input_buffer += event.unicode
                
                if self.selection:
                    if self.selection['type'] == 'node': self.nodes[self.selection['index']]['label'] = self.input_buffer
                    elif self.selection['type'] == 'edge':
                        try: self.edges[self.selection['index']]['weight'] = float(self.input_buffer)
                        except: pass
            else:
                if event.key == pygame.K_DELETE and self.selection:
                    if self.selection['type'] == 'node': self._delete_node(self.selection['index'])
                    else: self._delete_edge(self.selection['index'])
                    self.selection = None

    # --- LOGIC ---
    def _hit_test(self, pos):
        for i, n in enumerate(self.nodes):
            if np.linalg.norm(n['pos'] - pos) < 20 / self.zoom: return {'type': 'node', 'index': i}
        for i, e in enumerate(self.edges):
            u, v = self.nodes[e['u']]['pos'], self.nodes[e['v']]['pos']
            if self._dist_point_segment(pos, u, v) < 10 / self.zoom: return {'type': 'edge', 'index': i}
        return None

    def _dist_point_segment(self, p, a, b):
        ab = b - a
        if np.dot(ab, ab) == 0: return np.linalg.norm(p - a)
        t = max(0, min(1, np.dot(p - a, ab) / np.dot(ab, ab)))
        return np.linalg.norm(p - (a + t * ab))

    def _create_edge(self, u, v):
        for e in self.edges:
            if (e['u']==u and e['v']==v) or (e['u']==v and e['v']==u): return
        self.edges.append({'u': u, 'v': v, 'weight': 10.0})

    def _delete_node(self, idx):
        self.edges = [e for e in self.edges if e['u'] != idx and e['v'] != idx]
        for e in self.edges:
            if e['u'] > idx: e['u'] -= 1
            if e['v'] > idx: e['v'] -= 1
        self.nodes.pop(idx)

    def _delete_edge(self, idx):
        self.edges.pop(idx)

    def _handle_toolbar_click(self, pos):
        idx = int(pos[0] // 60)
        if 0 <= idx < len(self.tools):
            self.current_tool = self.tools[idx]['id']
            self.selection = None; self.editing_text = False; return
        if self.btn_run.collidepoint(pos): self.trigger_run = True
        elif self.btn_toggle_ids.collidepoint(pos): self.show_ids = not self.show_ids
        elif self.btn_toggle_w.collidepoint(pos): self.show_weights = not self.show_weights

    # --- DRAWING ---
    def draw(self):
        self.surface.fill(COLOR_BG)
        # Edges
        for i, e in enumerate(self.edges):
            u = tuple(self.world_to_screen(self.nodes[e['u']]['pos']).astype(int))
            v = tuple(self.world_to_screen(self.nodes[e['v']]['pos']).astype(int))
            is_sel = (self.selection and self.selection['type']=='edge' and self.selection['index']==i)
            col = COLOR_ACCENT if is_sel else (100, 100, 120)
            pygame.draw.line(self.surface, col, u, v, 4 if is_sel else 2)
            if self.show_weights: self._draw_label(((u[0]+v[0])//2, (u[1]+v[1])//2), str(int(e['weight'])), COLOR_PENDING)

        # Dragging Line
        if self.current_tool == TOOL_EDGE and self.drag_edge_start is not None:
            u = tuple(self.world_to_screen(self.nodes[self.drag_edge_start]['pos']).astype(int))
            pygame.draw.line(self.surface, COLOR_SELECTING, u, pygame.mouse.get_pos(), 2)

        # Nodes
        for i, n in enumerate(self.nodes):
            pos = tuple(self.world_to_screen(n['pos']).astype(int))
            is_sel = (self.selection and self.selection['type']=='node' and self.selection['index']==i)
            fill = COLOR_ACCENT if is_sel else (60, 120, 200)
            pygame.draw.circle(self.surface, fill, pos, 14)
            pygame.draw.circle(self.surface, COLOR_WHITE, pos, 14, 2)
            if self.show_ids: self._draw_text_centered(n['label'], pos)

        self._draw_ui()
        if self.selection: self._draw_inspector()
        return self.surface

    def _draw_label(self, pos, text, color):
        lbl = self.font.render(text, True, color)
        bg = pygame.Rect(pos[0], pos[1], lbl.get_width()+6, lbl.get_height()+4)
        bg.center = pos
        pygame.draw.rect(self.surface, (20, 20, 20), bg, border_radius=4)
        self.surface.blit(lbl, lbl.get_rect(center=pos))

    def _draw_text_centered(self, text, pos):
        lbl = self.bold_font.render(text, True, COLOR_WHITE)
        self.surface.blit(lbl, lbl.get_rect(center=pos))

    def _draw_ui(self):
        pygame.draw.rect(self.surface, COLOR_UI, (0, 0, self.width, self.toolbar_h))
        for i, tool in enumerate(self.tools):
            rect = pygame.Rect(i*60, 0, 60, self.toolbar_h)
            if self.current_tool == tool['id']: pygame.draw.rect(self.surface, COLOR_UI_ACTIVE, rect)
            txt = self.icon_font.render(tool['icon'], True, COLOR_TEXT)
            self.surface.blit(txt, txt.get_rect(center=rect.center))
            pygame.draw.line(self.surface, (30,30,30), (rect.right, 5), (rect.right, 45))
        self._draw_btn(self.btn_toggle_ids, "IDs", self.show_ids)
        self._draw_btn(self.btn_toggle_w, "Wgt", self.show_weights)
        
        hover = self.btn_run.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.surface, (0, 200, 100) if hover else (0, 160, 80), self.btn_run, border_radius=4)
        run_lbl = self.bold_font.render("RUN ▶", True, COLOR_WHITE)
        self.surface.blit(run_lbl, run_lbl.get_rect(center=self.btn_run.center))

    def _draw_btn(self, rect, text, state):
        col = COLOR_UI_ACTIVE if state else (60, 60, 60)
        pygame.draw.rect(self.surface, col, rect, border_radius=4)
        txt = self.font.render(text, True, COLOR_WHITE)
        self.surface.blit(txt, txt.get_rect(center=rect.center))

    def _draw_inspector(self):
        panel_rect = pygame.Rect(10, self.height - 110, 220, 100)
        pygame.draw.rect(self.surface, COLOR_UI, panel_rect, border_radius=8)
        pygame.draw.rect(self.surface, (100, 100, 100), panel_rect, 1, border_radius=8)
        
        type_str = "Node" if self.selection['type'] == 'node' else "Edge"
        title = self.bold_font.render(f"Edit {type_str}", True, COLOR_ACCENT)
        self.surface.blit(title, (panel_rect.x+10, panel_rect.y+10))
        
        box_rect = pygame.Rect(panel_rect.x+70, panel_rect.y+40, 130, 30)
        box_col = COLOR_WHITE if self.editing_text else (200, 200, 200)
        pygame.draw.rect(self.surface, (30, 30, 30), box_rect)
        pygame.draw.rect(self.surface, box_col, box_rect, 1)
        
        display_text = self.input_buffer if self.editing_text else \
            (self.nodes[self.selection['index']]['label'] if self.selection['type']=='node' else str(int(self.edges[self.selection['index']]['weight'])))
        txt_surf = self.font.render(display_text, True, COLOR_WHITE)
        self.surface.blit(txt_surf, (box_rect.x+5, box_rect.y+7))

    def export_data(self):
        if not self.nodes: return None, None, None, None
        
        positions = np.array([n['pos'] for n in self.nodes], dtype='f4')
        edge_list = [[e['u'], e['v'], e['weight']] for e in self.edges]
        edges_array = np.array(edge_list, dtype=np.float64) if edge_list else np.array([])
        
        line_geometry = np.empty((len(self.edges) * 2, 2), dtype='f4')
        for i, e in enumerate(self.edges):
            line_geometry[i*2] = positions[e['u']]
            line_geometry[i*2+1] = positions[e['v']]
            
        labels = [n['label'] for n in self.nodes]
        return edges_array, line_geometry, positions, labels