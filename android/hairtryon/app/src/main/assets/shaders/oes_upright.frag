#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
// Camera external texture -> upright RGB. uUvMat maps upright uv to camera-buffer uv
// (rotation, mirroring and the SurfaceTexture transform already folded in on the CPU).
precision highp float;
uniform samplerExternalOES uCam;
uniform mat3 uUvMat;
in vec2 vUv;
out vec4 oColor;
void main() {
    vec2 b = (uUvMat * vec3(vUv, 1.0)).xy;
    oColor = vec4(texture(uCam, b).rgb, 1.0);
}
