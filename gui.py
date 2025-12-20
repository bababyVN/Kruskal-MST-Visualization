import moderngl
import numpy as np
from pyrr import Matrix44
import pygame

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
    """Draws vertices with dynamic colors and sizes."""
    def __init__(self, ctx, vertices):
        self.ctx = ctx
        self.n_verts = len(vertices)
        
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec3 in_color;
                in float in_status; // 0.0 = Unselected, 1.0 = Selected
                out vec3 v_color;
                
                uniform mat4 u_transform;
                uniform float u_base_size;
                
                void main() {
                    gl_Position = u_transform * vec4(in_vert, 0.0, 1.0);
                    
                    if (in_status > 0.5) {
                        gl_PointSize = u_base_size * 1.5; 
                        v_color = in_color;
                    } else {
                        gl_PointSize = u_base_size * 0.6;
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

    def update_state(self, roots, statuses):
        new_colors = self.palette[roots]
        self.color_vbo.write(new_colors.tobytes())
        self.status_vbo.write(statuses.astype('f4').tobytes())

    def render(self, transform_matrix, point_size=12.0):
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.prog['u_transform'].write(transform_matrix)
        self.prog['u_base_size'].value = point_size
        self.vao.render(moderngl.POINTS)

class GraphRenderer:
    def __init__(self, ctx, n_edges, line_geometry, correct_aspect=False):
        self.ctx = ctx
        self.n_edges = n_edges
        self.zoom = 1.0
        self.offset = np.array([0.0, 0.0], dtype='f4')
        self.aspect_ratio = 1.0
        self.correct_aspect = correct_aspect # Store this setting
        
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
                void main() {
                    if (v_state > 2.9) {
                        f_color = vec4(0.0, 1.0, 0.2, 1.0); // Selecting
                    } else if (v_state > 1.9) {
                        f_color = vec4(1.0, 0.8, 0.0, 1.0); // MST
                    } else if (v_state > 0.9) {
                        discard; // Rejected
                    } else {
                        f_color = vec4(0.3, 0.3, 0.3, 0.2); // Unseen
                    }
                }
            '''
        )
        self.vbo = self.ctx.buffer(line_geometry.tobytes())
        self.state_data = np.zeros(n_edges * 2, dtype='f4') 
        self.state_vbo = self.ctx.buffer(self.state_data.tobytes(), dynamic=True)
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo, '2f', 'in_vert'),
            (self.state_vbo, '1f', 'in_state')
        ])
        self.current_matrix = None

    def update_camera(self):
        trans = Matrix44.from_translation([self.offset[0], self.offset[1], 0.0], dtype='f4')
        
        # Only apply aspect ratio correction if requested (for Limit Test)
        # For Editor data, we treat the screen as 1:1 to match pixel drawing
        if self.correct_aspect and self.aspect_ratio > 1.0:
            scale = Matrix44.from_scale([self.zoom / self.aspect_ratio, self.zoom, 1.0], dtype='f4')
        elif self.correct_aspect:
            scale = Matrix44.from_scale([self.zoom, self.zoom * self.aspect_ratio, 1.0], dtype='f4')
        else:
            # NO CORRECTION (Match Editor)
            scale = Matrix44.from_scale([self.zoom, self.zoom, 1.0], dtype='f4')
            
        self.current_matrix = (trans * scale).astype('f4').tobytes()
        self.prog['u_transform'].write(self.current_matrix)

    def update_states(self, start_idx, count, new_states_slice):
        offset = start_idx * 2 * 4 
        self.state_vbo.write(new_states_slice.tobytes(), offset=offset)

    def render(self):
        self.update_camera()
        self.vao.render(moderngl.LINES, vertices=self.n_edges * 2)