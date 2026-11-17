#version 300 es
// Draw an RGB texture, center-cropped to the target aspect ratio.
// Optional: flip Y (for CPU readback, which wants top row first) and a pink debug mask overlay.
precision highp float;
uniform sampler2D uTex;
uniform sampler2D uMask;     // smoothed hair probability in .r, top-row-first
uniform vec2 uCropScale;     // <= 1 on the axis that is cropped
uniform float uFlipY;
uniform float uOverlay;
in vec2 vUv;
out vec4 oColor;
void main() {
    vec2 uv = (vUv - 0.5) * uCropScale + 0.5;
    if (uFlipY > 0.5) uv.y = 1.0 - uv.y;
    vec3 c = texture(uTex, uv).rgb;
    if (uOverlay > 0.5) {
        float p = texture(uMask, vec2(uv.x, 1.0 - uv.y)).r;
        c = mix(c, vec3(1.0, 0.0, 0.31), 0.55 * p);
    }
    oColor = vec4(c, 1.0);
}
