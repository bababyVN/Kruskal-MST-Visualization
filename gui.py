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
    """Draws vertices."""
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
                uniform float u_zoom; 
                
                void main() {
                    gl_Position = u_transform * vec4(in_vert, 0.0, 1.0);
                    
                    // Scale point size with zoom, BUT CLAMP IT
                    // so it doesn't cover the screen when zoomed in x1000
                    float size = u_base_size * u_zoom;
                    float final_size = clamp(size, 2.0, 50.0); 
                    
                    gl_PointSize = final_size;
                    
                    if (in_status > 0.5) {
                        v_color = in_color;
                    } else {
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

    def render(self, transform_matrix, point_size=14.0, zoom=1.0):
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
        self.width = 1280.0
        self.height = 720.0
        
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
                
                void main() {
                    if (v_state > 2.9) {
                        f_color = vec4(0.0, 1.0, 0.2, 1.0); // Selecting (Green)
                    } else if (v_state > 1.9) {
                        f_color = vec4(1.0, 0.8, 0.0, 1.0); // MST (Gold)
                    } else if (v_state > 0.9) {
                        // Rejected (Red)
                        if (u_show_rejected < 0.5) discard;
                        f_color = vec4(1.0, 0.0, 0.0, 0.2); 
                    } else {
                        // Unseen (Grey)
                        if (u_show_unseen < 0.5) discard;
                        f_color = vec4(0.3, 0.3, 0.3, 0.15); 
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
        self.current_matrix = Matrix44.identity(dtype='f4').tobytes()
        
        # Set default visibility
        self.set_visibility(True, True)

    def set_camera(self, zoom, offset, w, h):
        self.zoom = zoom
        self.offset = offset
        self.width = w
        self.height = h
        self.update_camera()

    def set_visibility(self, show_rejected, show_unseen):
        self.prog['u_show_rejected'].value = 1.0 if show_rejected else 0.0
        self.prog['u_show_unseen'].value = 1.0 if show_unseen else 0.0

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
        self.prog['u_transform'].write(self.current_matrix)

    def update_states(self, start_idx, count, new_states_slice):
        offset = start_idx * 2 * 4 
        self.state_vbo.write(new_states_slice.tobytes(), offset=offset)

    def render(self):
        self.vao.render(moderngl.LINES, vertices=self.n_edges * 2)