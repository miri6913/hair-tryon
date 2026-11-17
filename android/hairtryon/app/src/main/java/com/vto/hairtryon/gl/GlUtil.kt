package com.vto.hairtryon.gl

import android.content.Context
import android.opengl.GLES30
import android.util.Log

private const val TAG = "VTO_GL"

object GlUtil {
    fun check(op: String) {
        val e = GLES30.glGetError()
        if (e != GLES30.GL_NO_ERROR) Log.e(TAG, "GL error 0x${e.toString(16)} after $op")
    }

    /** Reads a shader, expanding `#include "file.glsl"` lines from assets/shaders/ (GLSL ES has no include). */
    private fun readAsset(ctx: Context, path: String): String =
        ctx.assets.open(path).bufferedReader().use { it.readText() }.lines().joinToString("\n") { line ->
            val m = INCLUDE.matchEntire(line.trim())
            if (m != null) readAsset(ctx, "shaders/" + m.groupValues[1]) else line
        }

    private val INCLUDE = Regex("""#include\s+"([^"]+)"""")

    private fun compile(type: Int, src: String, name: String): Int {
        val s = GLES30.glCreateShader(type)
        GLES30.glShaderSource(s, src)
        GLES30.glCompileShader(s)
        val ok = IntArray(1)
        GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            val log = GLES30.glGetShaderInfoLog(s)
            GLES30.glDeleteShader(s)
            throw RuntimeException("Shader $name failed to compile: $log")
        }
        return s
    }

    /** Builds a program from `assets/shaders/<vert>` and `assets/shaders/<frag>`. */
    fun program(ctx: Context, vert: String, frag: String): Int {
        val vs = compile(GLES30.GL_VERTEX_SHADER, readAsset(ctx, "shaders/$vert"), vert)
        val fs = compile(GLES30.GL_FRAGMENT_SHADER, readAsset(ctx, "shaders/$frag"), frag)
        val p = GLES30.glCreateProgram()
        GLES30.glAttachShader(p, vs)
        GLES30.glAttachShader(p, fs)
        GLES30.glLinkProgram(p)
        val ok = IntArray(1)
        GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, ok, 0)
        if (ok[0] == 0) {
            val log = GLES30.glGetProgramInfoLog(p)
            GLES30.glDeleteProgram(p)
            throw RuntimeException("Program $vert+$frag failed to link: $log")
        }
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)
        return p
    }

    fun drawQuad() = GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
}

/** A linked GL program with cached uniform locations. */
class Program(val id: Int) {
    private val locs = HashMap<String, Int>()
    fun use() = GLES30.glUseProgram(id)
    private fun loc(name: String) = locs.getOrPut(name) { GLES30.glGetUniformLocation(id, name) }
    fun i(name: String, v: Int) = GLES30.glUniform1i(loc(name), v)
    fun f(name: String, v: Float) = GLES30.glUniform1f(loc(name), v)
    /** `uniform float name[n]` */
    fun fv(name: String, v: FloatArray) = GLES30.glUniform1fv(loc(name), v.size, v, 0)
    fun f2(name: String, v: FloatArray) = GLES30.glUniform2f(loc(name), v[0], v[1])
    fun f2(name: String, x: Float, y: Float) = GLES30.glUniform2f(loc(name), x, y)
    fun f3(name: String, x: Float, y: Float, z: Float) = GLES30.glUniform3f(loc(name), x, y, z)
    fun mat3(name: String, m: FloatArray) = GLES30.glUniformMatrix3fv(loc(name), 1, false, m, 0)

    /** Binds [tex] to texture unit [unit] and points sampler [name] at it. */
    fun tex(name: String, unit: Int, tex: Int, target: Int = GLES30.GL_TEXTURE_2D) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit)
        GLES30.glBindTexture(target, tex)
        GLES30.glUniform1i(loc(name), unit)
    }
}

/** A texture plus the framebuffer that renders into it. */
class Fbo(val width: Int, val height: Int, internalFormat: Int = GLES30.GL_RGBA8, linear: Boolean = true) {
    val tex: Int
    val fbo: Int

    init {
        val t = IntArray(1)
        GLES30.glGenTextures(1, t, 0)
        tex = t[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex)
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, internalFormat, width, height)
        // 32-bit float targets are not guaranteed to be linearly filterable, so they use NEAREST
        val filter = if (linear) GLES30.GL_LINEAR else GLES30.GL_NEAREST
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, filter)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, filter)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        val f = IntArray(1)
        GLES30.glGenFramebuffers(1, f, 0)
        fbo = f[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, tex, 0)
        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) throw RuntimeException("FBO incomplete: 0x${status.toString(16)}")
    }

    fun bind() {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glViewport(0, 0, width, height)
    }

    fun release() {
        GLES30.glDeleteFramebuffers(1, intArrayOf(fbo), 0)
        GLES30.glDeleteTextures(1, intArrayOf(tex), 0)
    }
}

/** 2D affine map x' = a x + c y + tx, y' = b x + d y + ty, composed with [then]. */
data class Affine(
    val a: Float = 1f, val b: Float = 0f, val c: Float = 0f, val d: Float = 1f,
    val tx: Float = 0f, val ty: Float = 0f,
) {
    /** Result applies `this` first, then [next]. */
    fun then(next: Affine) = Affine(
        a = next.a * a + next.c * b, b = next.b * a + next.d * b,
        c = next.a * c + next.c * d, d = next.b * c + next.d * d,
        tx = next.a * tx + next.c * ty + next.tx, ty = next.b * tx + next.d * ty + next.ty,
    )

    /** Column-major mat3 for GLSL. */
    fun toMat3() = floatArrayOf(a, b, 0f, c, d, 0f, tx, ty, 1f)

    companion object {
        fun translate(x: Float, y: Float) = Affine(tx = x, ty = y)

        /** Counter-clockwise rotation by [deg] (math convention, v up). */
        fun rotateCcw(deg: Int): Affine {
            val r = Math.toRadians(deg.toDouble())
            val cs = Math.cos(r).toFloat()
            val sn = Math.sin(r).toFloat()
            return Affine(a = cs, b = sn, c = -sn, d = cs)
        }

        val mirrorU = Affine(a = -1f, tx = 1f)

        /** From SurfaceTexture.getTransformMatrix (4x4 column-major). */
        fun fromSurfaceTexture(m: FloatArray) = Affine(a = m[0], b = m[1], c = m[4], d = m[5], tx = m[12], ty = m[13])
    }
}
