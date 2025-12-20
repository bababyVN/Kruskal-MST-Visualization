import moderngl
import numpy as np
from pyrr import Matrix44
import pygame

class TextureRenderer:
    """Renders 2D UI (Editor & Sliders) on top of the 3D world."""
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
    """Draws the vertices (nodes) as circles in Simulation Mode."""
    def __init__(self, ctx, vertices):
        self.ctx = ctx
        self.n_verts = len(vertices)
        
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                uniform mat4 u_transform;
                void main() {
                    gl_Position = u_transform * vec4(in_vert, 0.0, 1.0);
                    gl_PointSize = 10.0; // Fixed size dots
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 f_color;
                void main() {
                    vec2 circ = 2.0 * gl_PointCoord - 1.0;
                    if (dot(circ, circ) > 1.0) discard; // Make it round
                    f_color = vec4(0.2, 0.6, 1.0, 1.0); // Blue Nodes
                }
            '''
        )
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

    def render(self, transform_matrix):
        self.prog['u_transform'].write(transform_matrix)
        self.vao.render(moderngl.POINTS)

class GraphRenderer:
    """Draws the edges (lines)."""
    def __init__(self, ctx, n_edges, line_geometry):
        self.ctx = ctx
        self.n_edges = n_edges
        self.zoom = 1.0
        self.offset = np.array([0.0, 0.0], dtype='f4')
        
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
                    if (v_state > 1.9) f_color = vec4(0.0, 1.0, 0.0, 1.0);       // MST Green
                    else if (v_state > 0.9) f_color = vec4(1.0, 0.0, 0.0, 0.15); // Rejected Red
                    else f_color = vec4(0.5, 0.5, 0.5, 0.1);                     // Unseen Grey
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
        self.current_matrix = None # Store for CircleRenderer to use

    def update_camera(self):
        trans = Matrix44.from_translation([self.offset[0], self.offset[1], 0.0], dtype='f4')
        scale = Matrix44.from_scale([self.zoom, self.zoom, 1.0], dtype='f4')
        self.current_matrix = (trans * scale).astype('f4').tobytes()
        self.prog['u_transform'].write(self.current_matrix)

    def update_states(self, start_idx, count, new_states_slice):
        offset = start_idx * 2 * 4 
        self.state_vbo.write(new_states_slice.tobytes(), offset=offset)

    def render(self):
        self.update_camera()
        self.vao.render(moderngl.LINES, vertices=self.n_edges * 2)
