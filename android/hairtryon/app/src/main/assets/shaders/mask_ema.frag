#version 300 es
// Motion-adaptive temporal smoothing of the hair mask.
//   smooth = mix(previous, sigmoid(new logit), alpha),  alpha = clamp(uAlpha + uMotionGain * motion, 0, 1)
// "motion" is the change of the 5-tap neighborhood AVERAGE, not of the single pixel: real head movement shifts the
// whole boundary band (large change), while per-pixel edge noise averages out (small change). So noise is smoothed
// hard and genuine motion is followed quickly.
// uRaw: R16F logit(hair) - logit(background), top-row-first. uPrev: previous smoothed probability.
precision highp float;
uniform sampler2D uRaw;
uniform sampler2D uPrev;
uniform float uAlpha;
uniform float uMotionGain;
in vec2 vUv;
out vec4 oColor;
float sig(float x) { return 1.0 / (1.0 + exp(-x)); }
void main() {
    vec2 t = 1.5 / vec2(textureSize(uRaw, 0));
    float p = sig(texture(uRaw, vUv).r);
    float prev = texture(uPrev, vUv).r;
    float pAvg = 0.2 * (p
        + sig(texture(uRaw, vUv + vec2(t.x, 0.0)).r) + sig(texture(uRaw, vUv - vec2(t.x, 0.0)).r)
        + sig(texture(uRaw, vUv + vec2(0.0, t.y)).r) + sig(texture(uRaw, vUv - vec2(0.0, t.y)).r));
    float prevAvg = 0.2 * (prev
        + texture(uPrev, vUv + vec2(t.x, 0.0)).r + texture(uPrev, vUv - vec2(t.x, 0.0)).r
        + texture(uPrev, vUv + vec2(0.0, t.y)).r + texture(uPrev, vUv - vec2(0.0, t.y)).r);
    float a = clamp(uAlpha + uMotionGain * abs(pAvg - prevAvg), 0.0, 1.0);
    oColor = vec4(mix(prev, p, a), 0.0, 0.0, 1.0);
}
