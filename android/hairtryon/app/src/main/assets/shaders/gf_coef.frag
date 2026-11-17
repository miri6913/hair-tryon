#version 300 es
// Guided filter, step 2: from the box-filtered moments compute the linear coefficients q = a * I + b.
precision highp float;
uniform sampler2D uMoments;   // (mean I, mean p, mean I*p, mean I*I)
uniform float uEps;
in vec2 vUv;
out vec4 oColor;
void main() {
    vec4 m = texture(uMoments, vUv);
    float varI = max(m.w - m.x * m.x, 0.0);
    float covIp = m.z - m.x * m.y;
    float a = covIp / (varI + uEps);
    float b = m.y - a * m.x;
    oColor = vec4(a, b, 0.0, 1.0);
}
