package com.vto.hairtryon.ui.hairstyle

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.unit.dp
import com.vto.hairtryon.hairstyle.BlockPath

/**
 * The design's "block loader": four rounded squares hopping along one looping path, each a quarter loop apart.
 * The path (BlockPath, generated from the design's CSS keyframes) is in a 200x225 box; blocks are 30x30.
 */
@Composable
fun BlockLoader(colors: List<Color>, modifier: Modifier = Modifier) {
    val t by rememberInfiniteTransition(label = "blocks").animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween((BlockPath.SECONDS * 1000).toInt(), easing = LinearEasing), RepeatMode.Restart),
        label = "t",
    )
    val pose = remember { FloatArray(4) } // x, y, rotation, brightness of the block being drawn

    Canvas(modifier.size(200.dp, 225.dp)) {
        val side = 30.dp.toPx()
        for (i in colors.indices) {
            sample((t + BlockPath.PHASES[i]) % 1f, pose)
            val x = pose[0].dp.toPx() // resolved here: the transform lambda has no Density receiver
            val y = pose[1].dp.toPx()
            withTransform({
                translate(x, y)
                rotate(pose[2], pivot = Offset(side / 2, side / 2))
            }) {
                drawRoundRect(brighten(colors[i], pose[3]), size = Size(side, side), cornerRadius = CornerRadius(8.dp.toPx()))
            }
        }
    }
}

/** Linear interpolation along BlockPath.FRAMES at loop position [p] (0..1) into [out] = x, y, rotation, brightness. */
private fun sample(p: Float, out: FloatArray) {
    val f = BlockPath.FRAMES
    val n = f.size / BlockPath.STRIDE
    var lo = 0
    var hi = n - 1
    while (hi - lo > 1) { // last keyframe with t <= p
        val mid = (lo + hi) ushr 1
        if (f[mid * BlockPath.STRIDE] <= p) lo = mid else hi = mid
    }
    val a = lo * BlockPath.STRIDE
    val b = hi * BlockPath.STRIDE
    val span = f[b] - f[a]
    val k = if (span <= 0f) 0f else ((p - f[a]) / span).coerceIn(0f, 1f)
    for (j in 0 until 4) out[j] = f[a + 1 + j] + (f[b + 1 + j] - f[a + 1 + j]) * k
}

private fun brighten(c: Color, b: Float) =
    if (b <= 1f) c else Color((c.red * b).coerceAtMost(1f), (c.green * b).coerceAtMost(1f), (c.blue * b).coerceAtMost(1f), c.alpha)
