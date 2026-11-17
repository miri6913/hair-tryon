package com.vto.hairtryon.ui

import android.graphics.BitmapFactory
import android.net.Uri
import android.opengl.GLSurfaceView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import com.vto.hairtryon.camera.CameraSource
import com.vto.hairtryon.capture.CaptureSaver
import com.vto.hairtryon.gl.CameraRenderer
import com.vto.hairtryon.palette.Palette
import com.vto.hairtryon.palette.Shade
import com.vto.hairtryon.ui.components.CompareOverlay
import com.vto.hairtryon.ui.components.DesignIcons
import com.vto.hairtryon.ui.components.ModeTabs
import com.vto.hairtryon.ui.components.RoundIconButton
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/** What the Captured screen shows: the saved JPEG and the look that made it. */
class CapturedShot(val uri: Uri, val shade: Shade, val percent: Int)

private val MutedGrey = Color(0xFF8E8E93)
private val CameraCorner = 28.dp

@Composable
fun TryOnScreen(
    owner: LifecycleOwner,
    onBack: () -> Unit,
    onStyle: () -> Unit,
    onCaptured: (CapturedShot) -> Unit,
    stillPath: String? = null,
    initialRefine: Boolean = true,
    initialShade: String? = null,
    showDebug: Boolean = false,
) {
    val ctx = LocalContext.current
    val glView = remember {
        GLSurfaceView(ctx).apply {
            setEGLContextClientVersion(3)
            preserveEGLContextOnPause = false
            keepScreenOn = true // live camera preview: do not let the display time out
        }
    }
    val renderer = remember {
        CameraRenderer(ctx, glView).also {
            glView.setRenderer(it)
            glView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        }
    }
    val camera = remember { CameraSource(ctx) }

    var front by remember { mutableStateOf(true) }
    var shade by remember { mutableStateOf(Palette.byKey(initialShade) ?: Palette.shades[3]) }
    var intensity by remember { mutableFloatStateOf(0.8f) }
    var compare by remember { mutableStateOf(false) }
    var split by remember { mutableFloatStateOf(0.5f) }
    var showMask by remember { mutableStateOf(false) }
    var flipH by remember { mutableStateOf(false) }
    var refine by remember { mutableStateOf(initialRefine) }
    var prior by remember { mutableStateOf(false) }
    var fps by remember { mutableFloatStateOf(0f) }
    var segMs by remember { mutableFloatStateOf(0f) }
    var segFps by remember { mutableFloatStateOf(0f) }
    var status by remember { mutableStateOf("") }

    // push UI state into the renderer
    renderer.showMask = showMask
    renderer.flipH = flipH
    renderer.refine = refine
    renderer.usePrior = prior
    renderer.split = if (compare) split else 0f // left of the divider shows the original; captures force 0 in the renderer
    renderer.intensity = intensity
    renderer.shade = shade

    DisposableEffect(owner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> glView.onResume()
                Lifecycle.Event.ON_PAUSE -> glView.onPause()
                else -> {}
            }
        }
        owner.lifecycle.addObserver(observer)
        onDispose {
            owner.lifecycle.removeObserver(observer)
            camera.unbind() // the next screen (Hairstyle capture) binds its own camera use cases
            glView.onPause()
            renderer.release()
        }
    }
    if (stillPath == null) {
        LaunchedEffect(front) { camera.bind(owner, front, renderer) }
    } else {
        LaunchedEffect(stillPath) {
            val bmp = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(stillPath) }
            if (bmp == null) {
                status = "Cannot read $stillPath"
            } else {
                renderer.setStill(bmp)
                while (true) { // RENDERMODE_WHEN_DIRTY and no camera frames: drive ~30 Hz so the mask smoothing converges
                    glView.requestRender()
                    delay(33)
                }
            }
        }
    }
    LaunchedEffect(Unit) {
        while (true) {
            fps = renderer.fps
            segMs = renderer.segMs
            segFps = renderer.segFps
            delay(500)
        }
    }

    val capture = {
        status = "Saving..."
        renderer.requestCapture { bmp ->
            if (stillPath != null) {
                val mask = renderer.readSmoothedMask() // runs on the GL thread, inside the capture frame
                CaptureSaver.saveStill(ctx, bmp, mask) { f -> status = if (f != null) "Saved ${f.name}" else "Save failed" }
            } else {
                CaptureSaver.saveJpeg(ctx, bmp) { uri ->
                    glView.post {
                        if (uri != null) onCaptured(CapturedShot(uri, shade, (intensity * 100).toInt())) else status = "Save failed"
                    }
                }
            }
        }
    }

    Column(Modifier.fillMaxSize().background(Ink.Bg)) {
        // camera panel: rounded bottom corners, controls sit below it on solid black
        Box(
            Modifier.weight(1f).fillMaxWidth().pointerInput(compare) {
                if (compare) detectHorizontalDragGestures { change, _ -> split = (change.position.x / size.width).coerceIn(0.05f, 0.95f) }
            },
        ) {
            AndroidView(factory = { glView }, modifier = Modifier.fillMaxSize())
            RoundedBottomCorners(CameraCorner, Modifier.fillMaxSize())
            if (compare) CompareOverlay(split, lineHeight = null, handleY = 470.dp)

            // top bar: back, live pill, camera switch
            Column(Modifier.align(Alignment.TopCenter).fillMaxWidth().statusBarsPadding().padding(top = 8.dp, start = 16.dp, end = 16.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    RoundIconButton(DesignIcons.Back, "Back", onBack)
                    Row(
                        Modifier.clip(CircleShape).background(Ink.Glass).padding(horizontal = 12.dp, vertical = 7.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Box(Modifier.size(6.dp).clip(CircleShape).background(Ink.Green))
                        Text("Live · on device", style = Type.SmallBold)
                    }
                    RoundIconButton(DesignIcons.SwitchCamera, "Switch camera", { front = !front }, iconSize = 20.dp)
                }
                if (showDebug) {
                    Text(
                        "%.0f fps | seg %.0f ms @ %.0f/s".format(fps, segMs, segFps),
                        Modifier.fillMaxWidth().padding(top = 8.dp), style = Type.Chip, textAlign = TextAlign.Center,
                    )
                }
            }

            IntensityPill(intensity, { intensity = it }, Modifier.align(Alignment.TopEnd).padding(top = 170.dp, end = 12.dp))

            RoundIconButton(
                DesignIcons.Compare, "Compare before and after", { compare = !compare },
                Modifier.align(Alignment.BottomStart).padding(start = 16.dp, bottom = 16.dp),
                bg = if (compare) Ink.Text else Ink.Glass, tint = if (compare) Color.Black else Ink.Text, iconSize = 20.dp,
            )
        }

        Column(
            Modifier.fillMaxWidth().navigationBarsPadding().padding(top = 16.dp, bottom = 30.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.Bottom) {
                Text(shade.name, style = Type.Row)
                Text(shade.group, style = Type.Chip.copy(fontWeight = FontWeight.Normal, color = MutedGrey))
            }

            ShadeArc(shade, { shade = it }, capture)

            Text("Tap a shade · tap the centre to capture", style = Type.Chip.copy(fontWeight = FontWeight.Normal, color = MutedGrey))

            ModeTabs(styleSelected = false, onColor = {}, onStyle = onStyle)

            if (showDebug) {
                Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { refine = !refine }) { Text(if (refine) "Refine: on" else "Refine: off", color = Color.White) }
                    OutlinedButton(onClick = { prior = !prior }) { Text(if (prior) "Prior: on" else "Prior: off", color = Color.White) }
                    OutlinedButton(onClick = { showMask = !showMask }) { Text(if (showMask) "Mask on" else "Mask", color = Color.White) }
                    OutlinedButton(onClick = { flipH = !flipH }) { Text(if (flipH) "Flip: on" else "Flip", color = Color.White) }
                }
            }
            if (status.isNotEmpty()) Text(status, Modifier.fillMaxWidth(), style = Type.Chip.copy(color = Ink.Text2), textAlign = TextAlign.Center)
        }
    }
}

/**
 * Black corner pieces over the bottom corners of the camera. The camera is a SurfaceView, which does not take a rounded clip,
 * so the rounding is drawn on top of it (the surface sits behind the window; Compose draws above it).
 */
@Composable
private fun RoundedBottomCorners(radius: Dp, modifier: Modifier = Modifier) {
    Box(
        modifier.drawBehind {
            val r = radius.toPx()
            val w = size.width
            val h = size.height
            val left = Path().apply {
                moveTo(0f, h - r); lineTo(0f, h); lineTo(r, h)
                arcTo(Rect(0f, h - 2 * r, 2 * r, h), 90f, 90f, false)
                close()
            }
            val right = Path().apply {
                moveTo(w, h - r); lineTo(w, h); lineTo(w - r, h)
                arcTo(Rect(w - 2 * r, h - 2 * r, w, h), 90f, -90f, false)
                close()
            }
            drawPath(left, Color.Black)
            drawPath(right, Color.Black)
        },
    )
}

/** The design's vertical intensity control: 44 x 250 dp frosted pill, % on top, drop icon at the bottom. */
@Composable
private fun IntensityPill(value: Float, onChange: (Float) -> Unit, modifier: Modifier = Modifier) {
    val v = value.coerceIn(0f, 1f)
    Column(
        modifier
            .width(44.dp).height(250.dp)
            .clip(RoundedCornerShape(22.dp))
            .background(Color(0x731C1C1E)) // rgba(28,28,30,.45)
            .semantics { contentDescription = "Colour intensity ${(v * 100).toInt()} percent" }
            .padding(vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("${(v * 100).toInt()}", style = Type.Chip.copy(fontSize = 11.sp, fontWeight = FontWeight.Bold))
        Box(
            Modifier
                .weight(1f).width(24.dp)
                .pointerInput(Unit) { detectTapGestures { onChange((1f - it.y / size.height).coerceIn(0f, 1f)) } }
                .pointerInput(Unit) { detectVerticalDragGestures { change, _ -> onChange((1f - change.position.y / size.height).coerceIn(0f, 1f)) } }
                .drawBehind {
                    val trackW = 4.dp.toPx()
                    val x = (size.width - trackW) / 2
                    val round = CornerRadius(trackW / 2)
                    drawRoundRect(Color.White.copy(alpha = 0.25f), Offset(x, 0f), Size(trackW, size.height), round)
                    drawRoundRect(Ink.Text, Offset(x, size.height * (1f - v)), Size(trackW, size.height * v), round)
                    drawCircle(Color.White, radius = 10.dp.toPx(), center = Offset(size.width / 2, size.height * (1f - v)))
                },
        )
        Icon(DesignIcons.Drop, null, Modifier.size(16.dp), tint = Ink.Text)
    }
}

/**
 * The design's shade arc: the selected shade is the 84 dp capture button (white ring, shade-coloured disc); up to three shades on
 * each side get smaller and fainter, and tapping one selects it.
 */
@Composable
private fun ShadeArc(selected: Shade, onSelect: (Shade) -> Unit, onCapture: () -> Unit) {
    val shades = Palette.shades
    val sel = shades.indexOf(selected).coerceAtLeast(0)
    BoxWithConstraints(Modifier.fillMaxWidth().height(84.dp)) {
        val cx = maxWidth / 2
        // distance from the selected shade -> (offset from centre, size, opacity)
        val ring = listOf(Triple(66.dp, 40.dp, 0.9f), Triple(114.dp, 34.dp, 0.65f), Triple(156.dp, 28.dp, 0.4f))
        shades.forEachIndexed { j, s ->
            val d = j - sel
            val a = kotlin.math.abs(d)
            if (a == 0 || a > 3) return@forEachIndexed
            val (off, size, op) = ring[a - 1]
            Box(
                Modifier
                    .offset(x = cx + (if (d > 0) off else -off) - size / 2, y = 42.dp - size / 2)
                    .size(size)
                    .alpha(op)
                    .clip(CircleShape)
                    .background(Color(s.argb))
                    .border(1.dp, Color.White.copy(alpha = 0.16f), CircleShape)
                    .semantics { contentDescription = s.name }
                    .clickable(role = Role.Button) { onSelect(s) },
            )
        }
        Box(
            Modifier
                .align(Alignment.Center)
                .size(84.dp)
                .border(5.dp, Ink.Text, CircleShape)
                .clip(CircleShape)
                .semantics { contentDescription = "Take photo with ${selected.name}" }
                .clickable(role = Role.Button, onClick = onCapture),
            contentAlignment = Alignment.Center,
        ) {
            Box(Modifier.size(66.dp).clip(CircleShape).background(Color(selected.argb)))
        }
    }
}
