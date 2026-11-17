package com.vto.hairtryon.palette

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Expected values come from scripts/palette_lab_reference.py (OpenCV float RGB2LAB, as used in Phase 0).
 * OpenCV's float path uses lookup tables, so it differs from the closed-form math by up to ~0.2 (well under a
 * visible difference); the tolerance reflects that.
 */
class ColorMathTest {
    private fun check(rgb: Triple<Int, Int, Int>, l: Float, a: Float, b: Float) {
        val lab = ColorMath.rgbToLab(rgb.first, rgb.second, rgb.third)
        assertEquals("L of $rgb", l, lab[0], 0.35f)
        assertEquals("a of $rgb", a, lab[1], 0.35f)
        assertEquals("b of $rgb", b, lab[2], 0.35f)
    }

    @Test fun platinum() = check(Triple(235, 220, 190), 88.116f, 0.391f, 16.547f)
    @Test fun goldenBlonde() = check(Triple(214, 170, 90), 72.009f, 6.906f, 46.812f)
    @Test fun copper() = check(Triple(178, 82, 40), 46.436f, 36.453f, 41.938f)
    @Test fun burgundy() = check(Triple(110, 20, 45), 23.627f, 39.844f, 8.859f)
    @Test fun ashBlue() = check(Triple(90, 120, 175), 50.177f, 4.859f, -32.703f)
    @Test fun violet() = check(Triple(130, 70, 170), 41.211f, 43.812f, -43.719f)
    @Test fun emerald() = check(Triple(20, 130, 90), 48.065f, -39.250f, 13.734f)
    @Test fun black() = check(Triple(25, 20, 20), 6.927f, 2.219f, 0.797f)
    @Test fun white() = check(Triple(255, 255, 255), 100f, 0f, 0f)

    @Test fun lightnessMatchesLabL() {
        for ((r, g, b) in listOf(Triple(25, 20, 20), Triple(214, 170, 90), Triple(255, 255, 255))) {
            assertEquals(ColorMath.rgbToLab(r, g, b)[0], ColorMath.lightness(r, g, b), 0.1f)
        }
    }
}
