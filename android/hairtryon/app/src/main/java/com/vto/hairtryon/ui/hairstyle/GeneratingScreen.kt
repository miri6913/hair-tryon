package com.vto.hairtryon.ui.hairstyle

import android.graphics.Bitmap
import androidx.compose.animation.Crossfade
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.vto.hairtryon.hairstyle.HairApi
import com.vto.hairtryon.ui.components.DesignIcons
import com.vto.hairtryon.ui.components.PillButton
import com.vto.hairtryon.ui.components.RoundIconButton
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type
import kotlinx.coroutines.delay

private val CAPTIONS = listOf("Snip, snip. Almost there…", "Asking the stylist for a second opinion…", "Keeping your own colour…")

// "silver + honey" palette of the design: one honey block among three silver ones
private val LOADER_COLORS = listOf(Color(0xFFD2AC80), Color(0xFFD1D1D6), Color(0xFFD1D1D6), Color(0xFFD1D1D6))

/**
 * Hairstyle step 3: uploads the photo and waits. Leaving the screen (Cancel / back) cancels the request, because the call
 * runs in this composable's LaunchedEffect. On success [onDone] runs with session.result set; on failure session.error is set first.
 */
@Composable
fun GeneratingScreen(session: HairstyleSession, onCancel: () -> Unit, onDone: () -> Unit, onFailed: () -> Unit) {
    val photo = session.photo
    LaunchedEffect(Unit) {
        if (photo == null) return@LaunchedEffect onFailed()
        when (val r = HairApi.restyle(photo, session.styleId)) {
            is HairApi.Outcome.Ok -> {
                session.result = r.image
                session.seconds = r.seconds
                session.error = null
                onDone()
            }
            is HairApi.Outcome.Fail -> {
                session.error = r.kind
                onFailed()
            }
        }
    }

    var caption by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(3000)
            caption = (caption + 1) % CAPTIONS.size
        }
    }
    val backdrop = remember(photo) { photo?.let { blurry(it) } }

    Box(Modifier.fillMaxSize().background(Ink.Bg)) {
        if (backdrop != null) {
            Image(backdrop.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop, filterQuality = FilterQuality.Medium)
        }
        Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.66f)))

        RoundIconButton(DesignIcons.Close, "Cancel and go back", onCancel, Modifier.statusBarsPadding().padding(top = 8.dp, start = 16.dp))

        Column(Modifier.fillMaxWidth().padding(top = 200.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(28.dp)) {
            BlockLoader(LOADER_COLORS)
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Styling your hair", style = Type.Title)
                Box(Modifier.width(320.dp).height(22.dp), contentAlignment = Alignment.Center) {
                    Crossfade(caption, label = "caption") { Text(CAPTIONS[it], style = Type.Body15.copy(color = Ink.Text2), textAlign = TextAlign.Center) }
                }
            }
        }

        PillButton(
            "Cancel", onCancel, Modifier.align(Alignment.BottomCenter).fillMaxWidth().navigationBarsPadding().padding(start = 20.dp, end = 20.dp, bottom = 34.dp),
            fill = Ink.Surface2.copy(alpha = 0.8f),
        )
    }
}

/** Cheap blur: shrink to ~24 px wide and let the GPU's bilinear upscale smooth it (Modifier.blur needs API 31). */
private fun blurry(src: Bitmap): Bitmap {
    val w = 24
    return Bitmap.createScaledBitmap(src, w, (w * src.height.toFloat() / src.width).toInt().coerceAtLeast(1), true)
}
