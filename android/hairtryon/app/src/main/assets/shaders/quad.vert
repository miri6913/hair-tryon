#version 300 es
// Full-screen quad from gl_VertexID (draw with GL_TRIANGLE_STRIP, 4 vertices, no attributes).
// vUv: (0,0) bottom-left .. (1,1) top-right, matching FBO texture coordinates.
out vec2 vUv;
void main() {
    vec2 p = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
