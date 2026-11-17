package com.vto.hairtryon.ui.hairstyle

import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.vto.hairtryon.capture.CaptureSaver
import com.vto.hairtryon.hairstyle.Styles
import com.vto.hairtryon.ui.components.BeforeAfter
import com.vto.hairtryon.ui.components.DesignIcons
import com.vto.hairtryon.ui.components.PillButton
import com.vto.hairtryon.ui.components.RoundIconButton
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type

/** Hairstyle step 4: your photo vs the result with a draggable divider; Retake goes back to the camera, Save writes to Photos. */
@Composable
fun StyleResultScreen(session: HairstyleSession, onClose: () -> Unit, onRetake: () -> Unit) {
    val ctx = LocalContext.current
    val before = session.photo
    val after = session.result
    var savedUri by remember { mutableStateOf<Uri?>(null) }
    var saving by remember { mutableStateOf(false) }
    var compare by remember { mutableStateOf(true) }

    // Save and Share both need a file; the first of them writes it to Pictures/HairTryOn and the second reuses it
    fun withSavedUri(then: (Uri) -> Unit) {
        savedUri?.let { return then(it) }
        val bmp = after ?: return
        saving = true
        CaptureSaver.saveJpeg(ctx, bmp) { uri ->
            Handler(Looper.getMainLooper()).post {
                saving = false
                if (uri != null) { savedUri = uri; then(uri) }
            }
        }
    }

    Box(Modifier.fillMaxSize().background(Ink.Bg)) {
        if (before != null && after != null) BeforeAfter(before.asImageBitmap(), after.asImageBitmap(), handleY = 392.dp, compare = compare)

        Row(
            Modifier.fillMaxWidth().statusBarsPadding().padding(top = 8.dp, start = 16.dp, end = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            RoundIconButton(DesignIcons.Close, "Close", onClose)
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(1.dp)) {
                Text(Styles.byId(session.styleId)?.name ?: "", style = Type.Bar)
                Text("Made in ${"%.0f".format(session.seconds)} s · your colour kept", style = Type.Chip.copy(color = Ink.Text2))
            }
            RoundIconButton(DesignIcons.Share, "Share", { withSavedUri { ctx.startActivity(CaptureSaver.shareIntent(it)) } }, iconSize = 18.dp)
        }

        Row(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                .background(Brush.verticalGradient(0f to Color.Transparent, 0.4f to Color.Black.copy(alpha = 0.75f), 1f to Color.Black))
                .navigationBarsPadding().padding(start = 20.dp, end = 20.dp, top = 64.dp, bottom = 34.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            PillButton("Retake", onRetake, Modifier.weight(1f))
            PillButton(
                if (savedUri != null) "Saved" else if (saving) "Saving…" else "Save",
                { if (savedUri == null && !saving) withSavedUri { } }, Modifier.weight(1f), light = true,
                icon = if (savedUri != null) DesignIcons.Check else null,
            )
        }

        RoundIconButton(
            DesignIcons.Compare, "Compare before and after", { compare = !compare },
            Modifier.align(Alignment.BottomStart).navigationBarsPadding().padding(start = 20.dp, bottom = 112.dp),
            bg = if (compare) Ink.Text else Ink.Glass, tint = if (compare) Color.Black else Ink.Text, iconSize = 20.dp,
        )
    }
}
