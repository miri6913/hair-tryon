#version 300 es
// Hair recolor v2: GLSL port of recolor_hair2() in src/vto_spike/recolor2.py.
//  1. base lightness (masked blur) is mapped from the person's hair quantiles (uSrcQ, measured on the segmenter
//     thread, slope-capped on the CPU) onto the shade's real hair quantiles (uDstQ) with a monotone curve;
//  2. strand detail returns as a ratio to the base, faded near the mask edge and gated for near-black hair;
//  3. (a, b) follow the new lightness along the shade's shadow -> mid -> highlight ramp, per-channel gains;
//  4. lightness only changes in the solid interior, chroma reaches further into the soft edge (no light halo).
precision highp float;
#include "lab.glsl"
uniform sampler2D uRgb;     // upright camera frame
uniform sampler2D uMask;    // smoothed hair probability, top-row-first
uniform sampler2D uBlur;    // blurred (L*m, m), reduced resolution
uniform sampler2D uCoef;    // guided-filter (a, b)
uniform float uRefine;
uniform float uIntensity;   // 0..1
uniform float uSrcQ[5];     // person's hair lightness at p2 p10 p50 p90 p98 (after slope cap)
uniform float uDstQ[5];     // shade's lightness at the same quantiles
uniform vec2 uAbShadow;     // (a, b) at uDstQ[1]
uniform vec2 uAbMid;        // at uDstQ[2]
uniform vec2 uAbHigh;       // at uDstQ[3]
uniform vec2 uGain;         // chroma gain for a and b
uniform float uKeep;        // fraction of the photo's own chroma variation kept
uniform vec2 uMeanAB;       // mean (a, b) of the person's hair
uniform float uDetailGain;
uniform float uEps;
in vec2 vUv;
out vec4 oColor;

// piecewise-linear src -> dst, slope 1 outside [p2, p98]
float mapL(float x) {
    if (x <= uSrcQ[0]) return uDstQ[0] + (x - uSrcQ[0]);
    for (int i = 0; i < 4; i++) {
        if (x <= uSrcQ[i + 1]) {
            float t = (x - uSrcQ[i]) / max(uSrcQ[i + 1] - uSrcQ[i], 1e-3);
            return mix(uDstQ[i], uDstQ[i + 1], t);
        }
    }
    return uDstQ[4] + (x - uSrcQ[4]);
}

// shadow -> mid -> highlight ramp over new lightness, clamped like np.interp
vec2 ramp(float L) {
    float t1 = clamp((L - uDstQ[1]) / max(uDstQ[2] - uDstQ[1], 1e-3), 0.0, 1.0);
    float t2 = clamp((L - uDstQ[2]) / max(uDstQ[3] - uDstQ[2], 1e-3), 0.0, 1.0);
    return L <= uDstQ[2] ? mix(uAbShadow, uAbMid, t1) : mix(uAbMid, uAbHigh, t2);
}

void main() {
    vec3 rgb = texture(uRgb, vUv).rgb;
    float raw = refinedMask(uMask, uCoef, vUv, rgb, uRefine);
    float wL = smoothstep(0.5, 0.9, raw) * uIntensity;
    float wC = smoothstep(0.3, 0.8, raw) * uIntensity;
    if (max(wL, wC) < 0.002) {
        oColor = vec4(rgb, 1.0);
        return;
    }
    vec3 lab = rgbToLab(rgb);
    vec2 blur = texture(uBlur, vUv).rg;
    float base = blur.x / max(blur.y, 1e-4);

    float newBase = mapL(base);
    float ratio = clamp((lab.x + uEps) / (base + uEps), 0.4, 1.8);
    float gate = clamp((base - 3.0) / 12.0, 0.3, 1.0);   // near-black hair: detail is mostly noise
    float inner = smoothstep(0.55, 0.95, blur.y);        // fade the ratio to neutral near the mask boundary
    ratio = 1.0 + (ratio - 1.0) * inner * gate;

    float newL = (newBase + uEps) * pow(ratio, uDetailGain) - uEps;
    float over = max(newL - 88.0, 0.0);
    newL = clamp(min(newL, 88.0) + 12.0 * (1.0 - exp(-over / 12.0)), 0.0, 100.0);

    vec2 ab = ramp(newL) * uGain + uKeep * (lab.yz - uMeanAB);
    vec3 outLab = vec3(lab.x + (newL - lab.x) * wL, lab.yz + (ab - lab.yz) * wC);
    oColor = vec4(labToRgb(outLab), 1.0);
}
