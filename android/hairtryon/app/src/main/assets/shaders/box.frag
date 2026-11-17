#version 300 es
// One direction of a box filter of radius uRadius texels, on all four channels.
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uStep;       // one texel along the blur axis
uniform int uRadius;
in vec2 vUv;
out vec4 oColor;
void main() {
    vec4 acc = vec4(0.0);
    for (int i = -uRadius; i <= uRadius; i++) {
        acc += texture(uSrc, vUv + uStep * float(i));
    }
    oColor = acc / float(2 * uRadius + 1);
}
