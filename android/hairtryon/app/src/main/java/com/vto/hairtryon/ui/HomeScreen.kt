package com.vto.hairtryon.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.BiasAlignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.vto.hairtryon.R
import com.vto.hairtryon.ui.components.GlassChip
import com.vto.hairtryon.ui.components.PillButton
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type

/** Home, option B of the design: a "Yours | Honey" split photo, the pitch, and the two modes. */
@Composable
fun HomeScreen(onLiveColor: () -> Unit, onHairstyle: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Ink.Bg)) {
        Box(Modifier.fillMaxWidth().fillMaxHeight(0.735f)) {
            val focus = BiasAlignment(0f, -0.44f) // design: object-position 50% 28%
            Image(painterResource(R.drawable.home_yours), "Woman with long dark brown hair, her own colour", Modifier.fillMaxSize(), contentScale = ContentScale.Crop, alignment = focus)
            Image(
                painterResource(R.drawable.home_honey), "The same woman with honey brown hair",
                Modifier.fillMaxSize().drawWithContent { clipRect(left = size.width / 2) { this@drawWithContent.drawContent() } },
                contentScale = ContentScale.Crop, alignment = focus,
            )
            Box(Modifier.align(Alignment.Center).fillMaxHeight().width(1.5.dp).background(Ink.Text.copy(alpha = 0.85f)))
            Box(
                Modifier.fillMaxSize().background(
                    Brush.verticalGradient(
                        0f to Color.Black.copy(alpha = 0.4f), 0.2f to Color.Transparent, 0.58f to Color.Transparent, 1f to Color.Black,
                    ),
                ),
            )
        }

        Text("Try-On", Modifier.fillMaxWidth().statusBarsPadding().padding(top = 16.dp), style = Type.Row.copy(letterSpacing = 0.3.sp), textAlign = TextAlign.Center)
        GlassChip("Yours", Modifier.align(Alignment.TopStart).statusBarsPadding().padding(start = 20.dp, top = 64.dp))
        GlassChip("Honey", Modifier.align(Alignment.TopEnd).statusBarsPadding().padding(end = 20.dp, top = 64.dp))

        Column(
            Modifier.align(Alignment.BottomCenter).navigationBarsPadding().padding(start = 24.dp, end = 24.dp, bottom = 34.dp),
            verticalArrangement = Arrangement.spacedBy(28.dp),
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("See it before\nthe salon.", style = Type.Hero)
                Text("Change your colour live, or try a new cut from one photo.", style = Type.Body.copy(color = Ink.Text3))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                PillButton("Live Color", onLiveColor, Modifier.weight(1f), light = true, height = 56.dp)
                PillButton("Hairstyle", onHairstyle, Modifier.weight(1f), height = 56.dp)
            }
        }
    }
}
