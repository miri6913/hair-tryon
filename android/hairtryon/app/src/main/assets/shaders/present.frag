#version 300 es
// Final pass to the screen: center-crop, before/after split (left of uSplit = original), divider line.
precision highp float;
uniform sampler2D uOrig;
uniform sampler2D uResult;
uniform vec2 uCropScale;
uniform float uSplit;       // 0 = all recolored, 1 = all original
in vec2 vUv;
out vec4 oColor;
void main() {
    vec2 uv = (vUv - 0.5) * uCropScale + 0.5;
    vec3 c = vUv.x < uSplit ? texture(uOrig, uv).rgb : texture(uResult, uv).rgb;
    if (uSplit > 0.001 && uSplit < 0.999 && abs(vUv.x - uSplit) < 0.0025) c = vec3(1.0);
    oColor = vec4(c, 1.0);
}
