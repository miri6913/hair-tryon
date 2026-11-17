package com.vto.hairtryon.palette

import kotlin.math.cbrt
import kotlin.math.pow

/** sRGB (D65) <-> CIE Lab, matching OpenCV's float RGB2LAB used by the Python reference (src/vto_spike/recolor.py). */
object ColorMath {
    private const val XN = 0.950456
    private const val ZN = 1.088754

    private fun toLinear(c: Double) = if (c <= 0.04045) c / 12.92 else ((c + 0.055) / 1.055).pow(2.4)

    private fun f(t: Double) = if (t > 0.008856) cbrt(t) else 7.787 * t + 16.0 / 116.0

    /** [r], [g], [b] in 0..255. Returns (L 0..100, a, b). */
    fun rgbToLab(r: Int, g: Int, b: Int): FloatArray {
        val lr = toLinear(r / 255.0)
        val lg = toLinear(g / 255.0)
        val lb = toLinear(b / 255.0)
        val x = (0.412453 * lr + 0.357580 * lg + 0.180423 * lb) / XN
        val y = 0.212671 * lr + 0.715160 * lg + 0.072169 * lb
        val z = (0.019334 * lr + 0.119193 * lg + 0.950227 * lb) / ZN
        val fx = f(x)
        val fy = f(y)
        val fz = f(z)
        return floatArrayOf((116.0 * fy - 16.0).toFloat(), (500.0 * (fx - fy)).toFloat(), (200.0 * (fy - fz)).toFloat())
    }

    /** Approximate L* (0..100) for an 8-bit sRGB pixel; used for cheap hair-lightness statistics. */
    fun lightness(r: Int, g: Int, b: Int): Float {
        val y = 0.212671 * LIN[r] + 0.715160 * LIN[g] + 0.072169 * LIN[b]
        return (if (y > 0.008856) 116.0 * cbrt(y) - 16.0 else 903.3 * y).toFloat()
    }

    private val LIN = DoubleArray(256) { toLinear(it / 255.0) }
}
