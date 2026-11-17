package com.vto.hairtryon.ui.hairstyle

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.vto.hairtryon.ui.PhotoLoader
import com.vto.hairtryon.ui.components.ControlRow
import com.vto.hairtryon.ui.components.DesignIcons
import com.vto.hairtryon.ui.components.ModeTabs
import com.vto.hairtryon.ui.components.RoundIconButton
import com.vto.hairtryon.ui.components.Shutter
import com.vto.hairtryon.ui.components.SquareIconButton
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type
import java.util.concurrent.Executors

private const val TAG = "VTO_STYLE_CAPTURE"
private const val DESIGN_W = 390f // the canvas is 390 wide: the guide oval and the tips scale from it

/** Hairstyle step 1: a selfie viewfinder with a head guide. The server's face check (422) is the real validation. */
@Composable
fun StyleCaptureScreen(
    owner: LifecycleOwner,
    onClose: () -> Unit,
    onColor: () -> Unit,
    onLibrary: () -> Unit,
    onPhoto: (Bitmap) -> Unit,
) {
    val ctx = LocalContext.current
    val previewView = remember { PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }
    val imageCapture = remember { ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build() }
    val worker = remember { Executors.newSingleThreadExecutor() }
    var front by remember { mutableStateOf(true) }
    var busy by remember { mutableStateOf(false) }

    LaunchedEffect(front) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val selector = if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            try {
                provider.unbindAll()
                provider.bindToLifecycle(owner, selector, preview, imageCapture)
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
            }
        }, ContextCompat.getMainExecutor(ctx))
    }
    DisposableEffect(Unit) {
        onDispose {
            try {
                ProcessCameraProvider.getInstance(ctx).get().unbindAll()
            } catch (e: Exception) {
                Log.w(TAG, "unbind failed", e)
            }
            worker.shutdown()
        }
    }

    fun shoot() {
        if (busy) return
        busy = true
        imageCapture.takePicture(worker, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bmp = try {
                    upright(image, mirror = front)
                } finally {
                    image.close()
                }
                ContextCompat.getMainExecutor(ctx).execute { busy = false; onPhoto(bmp) }
            }

            override fun onError(e: ImageCaptureException) {
                Log.e(TAG, "capture failed", e)
                ContextCompat.getMainExecutor(ctx).execute { busy = false }
            }
        })
    }

    Box(Modifier.fillMaxSize().background(Ink.Bg)) {
        Column(Modifier.fillMaxSize()) {
            Box(Modifier.weight(1f).fillMaxWidth()) {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                BoxWithConstraints(Modifier.fillMaxSize()) {
                    val s = maxWidth.value / DESIGN_W
                    HeadGuide()
                    Row(
                        Modifier.fillMaxWidth().offset(y = (548 * s).dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
                    ) {
                        for (tip in listOf("One face", "Head in frame", "Good light")) Tip(tip)
                    }
                }
            }
            Column(
                Modifier.fillMaxWidth().background(Ink.Bg).navigationBarsPadding().padding(top = 20.dp, bottom = 30.dp),
                verticalArrangement = Arrangement.spacedBy(18.dp),
            ) {
                ModeTabs(styleSelected = true, onColor = onColor, onStyle = {}, modifier = Modifier.align(Alignment.CenterHorizontally))
                ControlRow(
                    left = { SquareIconButton(DesignIcons.Gallery, "Choose from photo library", onLibrary) },
                    center = { Shutter("Take photo", ::shoot) },
                    right = { RoundIconButton(DesignIcons.SwitchCamera, "Switch camera", { front = !front }, bg = Ink.Surface, size = 48.dp, iconSize = 20.dp) },
                )
            }
        }

        Row(
            Modifier.fillMaxWidth().statusBarsPadding().padding(top = 8.dp, start = 16.dp, end = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            RoundIconButton(DesignIcons.Close, "Close", onClose)
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text("Keep your whole head in frame", style = Type.Bar, textAlign = TextAlign.Center)
                Text("Leave a little space above your hair", style = Type.Small.copy(color = Ink.Text2), textAlign = TextAlign.Center)
            }
            Box(Modifier.width(44.dp))
        }
    }
}

/** Dimmed viewfinder with an oval hole (design: 270x390 at x 62..332, y 135..525, in a 390 wide frame) and a dotted outline. */
@Composable
private fun HeadGuide() {
    Canvas(Modifier.fillMaxSize()) {
        val k = size.width / DESIGN_W
        val oval = Rect(Offset(62f * k, 135f * k), Size(270f * k, 390f * k))
        val dim = Path().apply {
            fillType = PathFillType.EvenOdd
            addRect(Rect(Offset.Zero, size))
            addOval(oval)
        }
        drawPath(dim, Color.Black.copy(alpha = 0.58f))
        drawOval(
            Ink.Text, oval.topLeft, oval.size,
            style = Stroke(2.5.dp.toPx(), cap = StrokeCap.Round, pathEffect = PathEffect.dashPathEffect(floatArrayOf(0.5f, 9.dp.toPx()))),
        )
    }
}

@Composable
private fun Tip(text: String) {
    Row(
        Modifier.clip(CircleShape).background(Ink.Surface.copy(alpha = 0.7f)).padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Icon(DesignIcons.Check, null, Modifier.size(12.dp), tint = Ink.Green)
        Text(text, style = Type.Chip)
    }
}

/** The captured frame turned upright; the front camera is mirrored so the photo matches what the preview showed. Long side <= 2048. */
private fun upright(image: ImageProxy, mirror: Boolean): Bitmap {
    val raw = image.toBitmap()
    val m = Matrix().apply {
        postRotate(image.imageInfo.rotationDegrees.toFloat())
        if (mirror) postScale(-1f, 1f)
    }
    val turned = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
    return PhotoLoader.fit(turned, 2048)
}
