package com.vto.hairtryon.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * The divider drawn over a before/after view: a white line, a 36 dp handle and the Before / After pills.
 * [split] is the divider position as a fraction of the width. It only draws; the caller owns the drag gesture
 * (Live Color has the GL renderer do the actual split and drives this overlay, Result uses [BeforeAfter]). A null [lineHeight] runs the full height.
 */
@Composable
fun CompareOverlay(split: Float, lineHeight: Dp?, handleY: Dp, modifier: Modifier = Modifier) {
    BoxWithConstraints(modifier.fillMaxSize()) {
        val x = maxWidth * split
        Box(
            Modifier.offset(x - 1.dp).width(2.dp).then(if (lineHeight != null) Modifier.height(lineHeight) else Modifier.fillMaxSize())
                .background(Color.White.copy(alpha = 0.9f)),
        )
        Box(
            Modifier.offset(x - 18.dp, handleY).size(36.dp).clip(CircleShape).background(Color.White),
            contentAlignment = Alignment.Center,
        ) { Icon(DesignIcons.Handle, null, Modifier.size(18.dp), tint = Color.Black) }
        GlassChip("Before", Modifier.align(Alignment.TopStart).statusBarsPadding().padding(start = 20.dp, top = 60.dp))
        GlassChip("After", Modifier.align(Alignment.TopEnd).statusBarsPadding().padding(end = 20.dp, top = 60.dp))
    }
}

/** Two photos of the same size, shown whole (letterboxed); [after] shows to the right of the draggable divider. */
@Composable
fun BeforeAfter(before: ImageBitmap, after: ImageBitmap, modifier: Modifier = Modifier, handleY: Dp = 392.dp, compare: Boolean = true) {
    var split by remember { mutableFloatStateOf(0.5f) }
    Box(
        modifier.fillMaxSize().pointerInput(compare) {
            if (compare) detectHorizontalDragGestures { change, _ -> split = (change.position.x / size.width).coerceIn(0.05f, 0.95f) }
        },
    ) {
        // Fit, not Crop: a 3:4 photo on a tall phone screen would otherwise be cropped at the sides and zoomed ~1.7x (hair cut off)
        Image(before, "Before", Modifier.fillMaxSize(), contentScale = ContentScale.Fit, alignment = Alignment.Center)
        Image(
            after, "After",
            Modifier.fillMaxSize().drawWithContent { clipRect(left = if (compare) size.width * split else 0f) { this@drawWithContent.drawContent() } },
            contentScale = ContentScale.Fit, alignment = Alignment.Center,
        )
        if (compare) CompareOverlay(split, lineHeight = null, handleY = handleY)
    }
}
