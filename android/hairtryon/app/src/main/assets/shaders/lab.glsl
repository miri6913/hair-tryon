// sRGB (D65) <-> CIE Lab, same constants as OpenCV's RGB2LAB used by src/vto_spike/recolor.py.
// L in 0..100, a/b roughly -128..127. Include with: #include "lab.glsl"

const vec3 LAB_WHITE = vec3(0.950456, 1.0, 1.088754);

vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linearToSrgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

float labF(float t) { return t > 0.008856 ? pow(t, 1.0 / 3.0) : 7.787 * t + 16.0 / 116.0; }

float labFinv(float t) {
    float t3 = t * t * t;
    return t3 > 0.008856 ? t3 : (t - 16.0 / 116.0) / 7.787;
}

vec3 rgbToLab(vec3 rgb) {
    vec3 l = srgbToLinear(rgb);
    vec3 xyz = vec3(dot(l, vec3(0.412453, 0.357580, 0.180423)),
                    dot(l, vec3(0.212671, 0.715160, 0.072169)),
                    dot(l, vec3(0.019334, 0.119193, 0.950227))) / LAB_WHITE;
    float fx = labF(xyz.x), fy = labF(xyz.y), fz = labF(xyz.z);
    return vec3(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

vec3 labToRgb(vec3 lab) {
    float fy = (lab.x + 16.0) / 116.0;
    float fx = fy + lab.y / 500.0;
    float fz = fy - lab.z / 200.0;
    vec3 xyz = vec3(labFinv(fx), labFinv(fy), labFinv(fz)) * LAB_WHITE;
    vec3 l = vec3(dot(xyz, vec3(3.240479, -1.537150, -0.498535)),
                  dot(xyz, vec3(-0.969256, 1.875991, 0.041556)),
                  dot(xyz, vec3(0.055648, -0.204043, 1.057311)));
    return clamp(linearToSrgb(clamp(l, 0.0, 1.0)), 0.0, 1.0);
}

// Guided-filter output at this pixel: q = mean(a) * I + mean(b), with I the centered luma of the full-res frame.
// uCoef holds the box-filtered (a, b); falls back to the raw smoothed probability when refinement is off.
float refinedMask(sampler2D rawMask, sampler2D coef, vec2 uv, vec3 rgb, float useRefine) {
    float raw = texture(rawMask, vec2(uv.x, 1.0 - uv.y)).r;
    if (useRefine < 0.5) return raw;
    vec2 ab = texture(coef, uv).rg;
    float I = dot(rgb, vec3(0.299, 0.587, 0.114)) - 0.5;
    return clamp(ab.x * I + ab.y, 0.0, 1.0);
}

// Pull the soft guided-filter skirt in (port of mask_contrast in recolor.py).
float maskContrast(float p) {
    float t = clamp((p - 0.25) / 0.5, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
