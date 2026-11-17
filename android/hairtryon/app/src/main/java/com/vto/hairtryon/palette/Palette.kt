package com.vto.hairtryon.palette

/**
 * A hair shade as a measured distribution, not a single flat color (port of src/vto_spike/recolor2.py).
 *
 * [l] is the lightness of real hair of this shade at the 2/10/50/90/98th percentiles; the person's own hair
 * lightness is mapped onto it with a monotone curve, so their shading survives but lands in the shade's tonal range.
 * (a, b) follow the new lightness along shadow -> mid -> highlight ([abShadow] at l[1], [abMid] at l[2], [abHigh]
 * at l[3]); [gainA] / [gainB] scale red-green / yellow-blue separately (blondes get warmth without a pink cast).
 * [keepChroma] keeps a fraction of the photo's own chroma variation (reflected light), used for vivid shades.
 */
class Shade(
    val key: String,
    val name: String,
    val argb: Int, // swatch dot only
    val l: FloatArray,
    val abShadow: FloatArray,
    val abMid: FloatArray,
    val abHigh: FloatArray,
    val gainA: Float = 1f,
    val gainB: Float = 1f,
    val keepChroma: Float = 0f,
    val vivid: Boolean = false, // UI grouping only: "Natural" vs "Vivid"
) {
    val group: String get() = if (vivid) "Vivid" else "Natural"
}

object Palette {
    // Measured on 132 usable selfies, grouped by median hair lightness: scripts/hair_stats.py + build_shades.py.
    private class Tonal(val l: FloatArray, val shadow: FloatArray, val mid: FloatArray, val high: FloatArray)

    private val BLACK = Tonal(f(2.0f, 2.9f, 7.2f, 19.5f, 29.1f), f(2.1f, 1.1f), f(4.0f, 3.2f), f(5.5f, 6.0f))
    private val DARK_BROWN = Tonal(f(6.5f, 9.0f, 17.4f, 32.6f, 44.0f), f(4.4f, 4.1f), f(5.5f, 6.8f), f(5.8f, 8.9f))
    private val BROWN = Tonal(f(10.5f, 15.4f, 29.0f, 46.6f, 56.6f), f(6.1f, 8.1f), f(6.5f, 11.2f), f(6.1f, 12.9f))
    private val LIGHT_BROWN = Tonal(f(20.6f, 26.8f, 42.2f, 62.3f, 71.7f), f(6.9f, 10.8f), f(6.3f, 12.7f), f(5.1f, 13.2f))

    private fun f(vararg v: Float) = v

    private fun natural(key: String, name: String, argb: Int, t: Tonal, gainA: Float = 1f, gainB: Float = 1f) =
        Shade(key, name, argb, t.l, t.shadow, t.mid, t.high, gainA, gainB)

    /** A fashion color follows a natural tonal distribution: stays saturated in shadow, desaturates toward white. */
    private fun vivid(key: String, name: String, r: Int, g: Int, b: Int, t: Tonal): Shade {
        val lab = ColorMath.rgbToLab(r, g, b)
        val a0 = lab[1]
        val b0 = lab[2]
        return Shade(
            key, name, (0xFF shl 24) or (r shl 16) or (g shl 8) or b, t.l,
            f(a0 * 0.95f, b0 * 0.95f), f(a0 * 0.85f, b0 * 0.85f), f(a0 * 0.5f, b0 * 0.5f),
            keepChroma = 0.15f, vivid = true,
        )
    }

    // Warm gains are tuned by eye on results/review sheets: b (yellow) up much more than a (red) avoids a pink cast.
    val shades = listOf(
        natural("black", "Black", 0xFF1E1A1A.toInt(), BLACK),
        natural("dark_brown", "Dark brown", 0xFF3B2A20.toInt(), DARK_BROWN),
        natural("chestnut", "Chestnut", 0xFF6B4630.toInt(), BROWN, 1.05f, 1.35f),
        natural("honey", "Honey", 0xFF9C7348.toInt(), LIGHT_BROWN, 1.0f, 1.6f),
        vivid("copper", "Copper", 178, 82, 40, LIGHT_BROWN),
        vivid("burgundy", "Burgundy", 120, 25, 50, DARK_BROWN),
        vivid("blue", "Blue", 40, 90, 200, LIGHT_BROWN),
        vivid("violet", "Violet", 125, 60, 170, LIGHT_BROWN),
        vivid("emerald", "Emerald", 20, 140, 95, LIGHT_BROWN),
    )

    fun byKey(key: String?): Shade? = shades.firstOrNull { it.key == key }

    /**
     * Source quantiles of the person's hair lightness, adjusted so the map to [dst] never gets steeper than
     * [maxSlope] (port of _monotone_map in recolor2.py): near-flat dark hair must not turn noise into blotches.
     * The median is the anchor; gaps are widened outward from it.
     */
    fun capSlope(src: FloatArray, dst: FloatArray, maxSlope: Float = 3f): FloatArray {
        val s = src.copyOf()
        for (i in 3..4) s[i] = maxOf(s[i], s[i - 1] + maxOf((dst[i] - dst[i - 1]) / maxSlope, 0.5f))
        for (i in 1 downTo 0) s[i] = minOf(s[i], s[i + 1] - maxOf((dst[i + 1] - dst[i]) / maxSlope, 0.5f))
        return s
    }
}
