#version 300 es
// One direction of a separable Gaussian (13 taps). uStep = one texel along the blur axis.
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uStep;
uniform float uSigma;
in vec2 vUv;
out vec4 oColor;
void main() {
    vec4 acc = vec4(0.0);
    float ws = 0.0;
    for (int i = -6; i <= 6; i++) {
        float w = exp(-float(i * i) / (2.0 * uSigma * uSigma));
        acc += texture(uSrc, vUv + uStep * float(i)) * w;
        ws += w;
    }
    oColor = acc / ws;
}
