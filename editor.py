import pygame
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import math
from config import *

class GraphEditor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Data
        self.nodes = [] 
        self.edges = [] 
        
        # Rendering State
        self.dirty = True  # Flag to tell Main to update GPU buffers
        
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
        
        # Save Popup
        self.show_save_menu = False
        
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
        
        # Buttons
        btn_y = 10
        self.btn_run = pygame.Rect(width - 70, btn_y, 60, 30)
        self.btn_load = pygame.Rect(width - 140, btn_y, 60, 30)
        self.btn_save = pygame.Rect(width - 210, btn_y, 60, 30)
        
        self.save_menu_rect = pygame.Rect(width - 210, btn_y + 35, 80, 60)
        self.btn_save_xy = pygame.Rect(self.save_menu_rect.x, self.save_menu_rect.y, 80, 30)
        self.btn_save_topo = pygame.Rect(self.save_menu_rect.x, self.save_menu_rect.y + 30, 80, 30)
        
        self.btn_toggle_ids = pygame.Rect(width - 300, 12, 60, 26)
        self.btn_toggle_w = pygame.Rect(width - 370, 12, 60, 26)
        
        self.btn_gen = pygame.Rect(width - 440, 12, 60, 26)
        
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
        
        if len(self.edges) < 5000:
            self.hover_item = self._hit_test(self.mouse_world)
        else:
            self.hover_item = None
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.show_save_menu:
                    if self.btn_save_xy.collidepoint(m_pos):
                        self._save_graph(with_coords=True); self.show_save_menu = False; return
                    elif self.btn_save_topo.collidepoint(m_pos):
                        self._save_graph(with_coords=False); self.show_save_menu = False; return
                    elif not self.save_menu_rect.collidepoint(m_pos) and not self.btn_save.collidepoint(m_pos):
                        self.show_save_menu = False
                
                if m_pos[1] < self.toolbar_h:
                    self._handle_toolbar_click(m_pos); return
                
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
                    base_id = len(self.nodes)
                    new_label = str(base_id)
                    while any(n['label'] == new_label for n in self.nodes): base_id += 1; new_label = str(base_id)
                    self.nodes.append({'pos': self.mouse_world, 'label': new_label})
                    self.dirty = True 
                    
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
                self.dirty = True 

        elif event.type == pygame.MOUSEWHEEL:
            zoom_factor = 1.1 if event.y > 0 else 0.9
            world_before = self.screen_to_world(m_pos)
            self.zoom = max(0.0001, self.zoom * zoom_factor)
            self.offset += (np.array(m_pos) - self.world_to_screen(world_before))

        elif event.type == pygame.KEYDOWN:
            if self.editing_text:
                if event.key == pygame.K_RETURN: self._confirm_edit(); self.editing_text = False
                elif event.key == pygame.K_BACKSPACE: self.input_buffer = self.input_buffer[:-1]
                else: self.input_buffer += event.unicode
            else:
                if event.key == pygame.K_DELETE and self.selection:
                    if self.selection['type'] == 'node': self._delete_node(self.selection['index'])
                    else: self._delete_edge(self.selection['index'])
                    self.selection = None
                elif event.key == pygame.K_g:
                    self._prompt_random_graph()

    def _confirm_edit(self):
        if not self.selection: return
        if self.selection['type'] == 'node':
            new_label = self.input_buffer.strip()
            is_dup = False
            curr = self.selection['index']
            for i, n in enumerate(self.nodes):
                if i != curr and n['label'] == new_label: is_dup = True; break
            if is_dup: 
                print(f"ID '{new_label}' exists!"); self.input_buffer = self.nodes[curr]['label']
            else: self.nodes[curr]['label'] = new_label
        elif self.selection['type'] == 'edge':
            try: self.edges[self.selection['index']]['weight'] = float(self.input_buffer)
            except: pass

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
        self.dirty = True

    def _delete_node(self, idx):
        self.edges = [e for e in self.edges if e['u'] != idx and e['v'] != idx]
        for e in self.edges:
            if e['u'] > idx: e['u'] -= 1
            if e['v'] > idx: e['v'] -= 1
        self.nodes.pop(idx)
        self.dirty = True

    def _delete_edge(self, idx):
        self.edges.pop(idx)
        self.dirty = True

    def _handle_toolbar_click(self, pos):
        idx = int(pos[0] // 60)
        if 0 <= idx < len(self.tools):
            self.current_tool = self.tools[idx]['id']
            self.selection = None; self.editing_text = False; return
        if self.btn_run.collidepoint(pos): self.trigger_run = True
        elif self.btn_toggle_ids.collidepoint(pos): self.show_ids = not self.show_ids
        elif self.btn_toggle_w.collidepoint(pos): self.show_weights = not self.show_weights
        elif self.btn_save.collidepoint(pos): self.show_save_menu = not self.show_save_menu
        elif self.btn_load.collidepoint(pos): self._load_graph(); self.show_save_menu = False
        elif self.btn_gen.collidepoint(pos): self._prompt_random_graph()

    def _prompt_random_graph(self):
        try:
            n = simpledialog.askinteger("Random Graph", "Number of Nodes (N):", parent=self.root, minvalue=2, maxvalue=5000, initialvalue=20)
            if not n: return
            max_edges = n * (n - 1) // 2
            m = simpledialog.askinteger("Random Graph", f"Number of Edges (M) [Max {max_edges}]:", parent=self.root, minvalue=1, maxvalue=max_edges, initialvalue=30)
            if not m: return
            
            use_random = messagebox.askyesno(
                "Layout Generation", 
                "Generate with RANDOM positions?\n\n(Click 'No' for Circular/Topo Layout)"
            )
            
            self._generate_random_graph(n, m, use_random)
        except Exception as e:
            print(f"Generation error: {e}")

    def _generate_random_graph(self, n, m, use_random):
        print(f"Generating {n} nodes, {m} edges (Random: {use_random})...")
        self.nodes = []
        self.edges = []
        
        if use_random:
            area_scale = math.sqrt(n) * 50
            center_x, center_y = 0, 0
            for i in range(n):
                x = np.random.uniform(center_x - area_scale, center_x + area_scale)
                y = np.random.uniform(center_y - area_scale, center_y + area_scale)
                self.nodes.append({'pos': np.array([x, y], dtype='f4'), 'label': str(i)})
        else:
            radius = max(200, math.sqrt(n) * 100)
            for i in range(n):
                angle = (2 * math.pi * i) / n
                x = math.cos(angle) * radius
                y = math.sin(angle) * radius
                self.nodes.append({'pos': np.array([x, y], dtype='f4'), 'label': str(i)})
            
        existing_edges = set()
        edges_created = 0
        attempts = 0
        max_attempts = m * 5 
        
        while edges_created < m and attempts < max_attempts:
            attempts += 1
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v: continue
            key = tuple(sorted((u, v)))
            if key in existing_edges: continue
            existing_edges.add(key)
            w = np.random.randint(1, 100)
            self.edges.append({'u': u, 'v': v, 'weight': float(w)})
            edges_created += 1
            
        self.selection = None
        self.dirty = True 
        print("Graph generation complete.")

    def _save_graph(self, with_coords):
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if not filename: return
        try:
            with open(filename, 'w') as f:
                for e in self.edges:
                    u, v = self.nodes[e['u']], self.nodes[e['v']]
                    line = f"{u['label']} {v['label']} {e['weight']:.2f}"
                    if with_coords: line += f" {u['pos'][0]:.2f} {u['pos'][1]:.2f} {v['pos'][0]:.2f} {v['pos'][1]:.2f}"
                    f.write(line + "\n")
            print(f"Saved {filename}")
        except Exception as e: print(e)

    def _load_graph(self):
        filename = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not filename: return
        print(f"Loading {filename}...")
        try:
            with open(filename, 'r') as f: lines = f.readlines()
            
            has_coords = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 7:
                    has_coords = True
                    break
            
            use_random_layout = False
            if not has_coords and len(lines) > 0:
                use_random_layout = messagebox.askyesno(
                    "Layout Selection", 
                    "No coordinates found in file.\n\nGenerate RANDOM positions?\n(Click 'No' for Circular Layout)"
                )

            new_nodes = []
            new_edges = []
            node_map = {}
            existing_edges = set()
            
            def get_node(label, x=None, y=None):
                if label not in node_map:
                    pos = np.array([x if x else 0, y if y else 0], dtype='f4')
                    node_map[label] = len(new_nodes)
                    new_nodes.append({'pos': pos, 'label': label})
                return node_map[label]

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: continue
                try: w = float(parts[2])
                except: continue
                
                if len(parts) >= 7:
                    u = get_node(parts[0], float(parts[3]), float(parts[4]))
                    new_nodes[u]['pos'] = np.array([float(parts[3]), float(parts[4])], dtype='f4')
                    v = get_node(parts[1], float(parts[5]), float(parts[6]))
                    new_nodes[v]['pos'] = np.array([float(parts[5]), float(parts[6])], dtype='f4')
                else:
                    u, v = get_node(parts[0]), get_node(parts[1])
                
                key = tuple(sorted((u, v)))
                if key not in existing_edges and u != v:
                    existing_edges.add(key)
                    new_edges.append({'u': u, 'v': v, 'weight': w})
            
            if not has_coords:
                n = len(new_nodes)
                if use_random_layout:
                    area_scale = math.sqrt(n) * 50
                    for node in new_nodes:
                        rx = np.random.uniform(-area_scale, area_scale)
                        ry = np.random.uniform(-area_scale, area_scale)
                        node['pos'] = np.array([rx, ry], dtype='f4')
                else:
                    rad = max(200, math.sqrt(n)*100)
                    for i, node in enumerate(new_nodes):
                        a = (2*math.pi*i)/n if n>0 else 0
                        node['pos'] = np.array([math.cos(a)*rad, math.sin(a)*rad], dtype='f4')

            self.nodes = new_nodes
            self.edges = new_edges
            self.selection = None
            self.dirty = True
            print(f"Loaded {len(self.nodes)} nodes, {len(self.edges)} edges.")
        except Exception as e: print(e)

    def draw(self, draw_graph=True):
        if draw_graph: self.surface.fill(COLOR_BG) 
        else: self.surface.fill((0, 0, 0, 0))

        if draw_graph:
            for i, e in enumerate(self.edges):
                u = tuple(self.world_to_screen(self.nodes[e['u']]['pos']).astype(int))
                v = tuple(self.world_to_screen(self.nodes[e['v']]['pos']).astype(int))
                if not (0<=u[0]<=self.width or 0<=v[0]<=self.width): continue
                if not (0<=u[1]<=self.height or 0<=v[1]<=self.height): continue
                is_sel = (self.selection and self.selection['type']=='edge' and self.selection['index']==i)
                col = COLOR_ACCENT if is_sel else (100, 100, 120)
                pygame.draw.line(self.surface, col, u, v, 4 if is_sel else 2)
            
            for i, n in enumerate(self.nodes):
                pos = tuple(self.world_to_screen(n['pos']).astype(int))
                if not (-20<=pos[0]<=self.width+20 and -20<=pos[1]<=self.height+20): continue
                is_sel = (self.selection and self.selection['type']=='node' and self.selection['index']==i)
                fill = COLOR_ACCENT if is_sel else (60, 120, 200)
                pygame.draw.circle(self.surface, fill, pos, 14)
                pygame.draw.circle(self.surface, COLOR_WHITE, pos, 14, 2)
        
        # --- TEXT RENDERING WITH CULLING ---
        # UPDATED: Checks BOTH vertices AND edges count against threshold
        if len(self.nodes) > TEXT_RENDER_THRESHOLD or len(self.edges) > TEXT_RENDER_THRESHOLD:
            if self.show_ids or self.show_weights:
                warn_text = "Graph is too big - ID and Weight rendering disabled"
                lbl = self.bold_font.render(warn_text, True, (255, 80, 80))
                bg = pygame.Rect(0, 0, lbl.get_width() + 20, lbl.get_height() + 10)
                bg.center = (self.width // 2, self.height - 40)
                pygame.draw.rect(self.surface, (30, 30, 30), bg, border_radius=5)
                pygame.draw.rect(self.surface, (255, 80, 80), bg, 1, border_radius=5)
                self.surface.blit(lbl, lbl.get_rect(center=bg.center))
        else:
            if self.show_weights:
                for i, e in enumerate(self.edges):
                    u_pos = self.nodes[e['u']]['pos']
                    v_pos = self.nodes[e['v']]['pos']
                    mid_pos = (u_pos + v_pos) * 0.5
                    mid_scr = self.world_to_screen(mid_pos).astype(int)
                    if not (0 <= mid_scr[0] <= self.width and 0 <= mid_scr[1] <= self.height): continue
                    self._draw_label(tuple(mid_scr), str(int(e['weight'])), COLOR_PENDING)

            if self.show_ids:
                for i, n in enumerate(self.nodes):
                    pos = self.world_to_screen(n['pos']).astype(int)
                    if not (0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height): continue
                    self._draw_text_centered(n['label'], tuple(pos))

        if self.current_tool == TOOL_EDGE and self.drag_edge_start is not None:
            u = tuple(self.world_to_screen(self.nodes[self.drag_edge_start]['pos']).astype(int))
            pygame.draw.line(self.surface, COLOR_SELECTING, u, pygame.mouse.get_pos(), 2)

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
        self._draw_btn(self.btn_save, "SAVE", self.show_save_menu)
        self._draw_btn(self.btn_load, "LOAD", False)
        self._draw_btn(self.btn_gen, "GEN", False)
        
        if self.show_save_menu:
            pygame.draw.rect(self.surface, COLOR_UI, self.save_menu_rect, border_radius=5)
            pygame.draw.rect(self.surface, (100,100,100), self.save_menu_rect, 1, border_radius=5)
            self._draw_btn(self.btn_save_xy, "XY Data", False)
            self._draw_btn(self.btn_save_topo, "Topo Only", False)

        hover = self.btn_run.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.surface, (0, 200, 100) if hover else (0, 160, 80), self.btn_run, border_radius=4)
        run_lbl = self.bold_font.render("RUN", True, COLOR_WHITE)
        self.surface.blit(run_lbl, run_lbl.get_rect(center=self.btn_run.center))

    def _draw_btn(self, rect, text, state):
        col = COLOR_UI_ACTIVE if state else (60, 60, 60)
        if rect.collidepoint(pygame.mouse.get_pos()) and not state: col = COLOR_UI_HOVER
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
        u_indices = [e['u'] for e in self.edges]
        v_indices = [e['v'] for e in self.edges]
        weights = [e['weight'] for e in self.edges]
        edges_array = np.column_stack((u_indices, v_indices, weights)).astype(np.float64)
        if len(self.edges) > 0:
            start_points = positions[u_indices]
            end_points = positions[v_indices]
            line_geometry = np.empty((len(self.edges) * 2, 2), dtype='f4')
            line_geometry[0::2] = start_points
            line_geometry[1::2] = end_points
        else:
            line_geometry = np.array([], dtype='f4')
        labels = [n['label'] for n in self.nodes]
        return edges_array, line_geometry, positions, labels