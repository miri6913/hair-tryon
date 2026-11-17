#version 300 es
// Guided filter, step 1: pack (I, p, I*p, I*I) with I = centered luma of the camera frame, p = hair probability.
// Runs at 1/4 resolution (the standard "fast guided filter" subsampling), so I is a 4x4 box average of the
// full-res luma, taken with four bilinear taps. 32-bit float target: variance = E[I^2] - E[I]^2 needs the precision.
precision highp float;
uniform sampler2D uRgb;
uniform sampler2D uMask;   // smoothed hair probability, top-row-first
uniform vec2 uTexel;       // one texel of uRgb
in vec2 vUv;
out vec4 oColor;
float luma(vec2 uv) { return dot(texture(uRgb, uv).rgb, vec3(0.299, 0.587, 0.114)); }
void main() {
    vec2 d = uTexel;
    float I = 0.25 * (luma(vUv + vec2(-d.x, -d.y)) + luma(vUv + vec2(d.x, -d.y)) +
                      luma(vUv + vec2(-d.x, d.y)) + luma(vUv + vec2(d.x, d.y))) - 0.5;
    float p = texture(uMask, vec2(vUv.x, 1.0 - vUv.y)).r;
    oColor = vec4(I, p, I * p, I * I);
}
