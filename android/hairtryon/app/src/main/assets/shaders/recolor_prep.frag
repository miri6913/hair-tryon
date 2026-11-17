#version 300 es
// Pre-pass for the masked base lightness: writes (L * m, m) at reduced resolution; two blur passes follow,
// and base = blur(L*m) / blur(m) as in recolor.py (_masked_blur).
precision highp float;
#include "lab.glsl"
uniform sampler2D uRgb;
uniform sampler2D uMask;   // smoothed hair probability, top-row-first
uniform sampler2D uCoef;   // guided-filter (a, b)
uniform float uRefine;
in vec2 vUv;
out vec4 oColor;
void main() {
    vec3 rgb = texture(uRgb, vUv).rgb;
    float m = maskContrast(refinedMask(uMask, uCoef, vUv, rgb, uRefine));
    oColor = vec4(rgbToLab(rgb).x * m, m, 0.0, 1.0);
}
