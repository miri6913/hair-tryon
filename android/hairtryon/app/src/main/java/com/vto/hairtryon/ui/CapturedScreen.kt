package com.vto.hairtryon.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.vto.hairtryon.capture.CaptureSaver
import com.vto.hairtryon.ui.components.DesignIcons
import com.vto.hairtryon.ui.components.PillButton
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** After the shutter in Live Color: "Saved to Photos", the picture with a shade chip, Share and Try another shade. */
@Composable
fun CapturedScreen(shot: CapturedShot, onDone: () -> Unit, onAnother: () -> Unit) {
    val ctx = LocalContext.current
    var image by remember { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(shot.uri) {
        image = withContext(Dispatchers.IO) { PhotoLoader.load(ctx, shot.uri, 1600) }
    }
    Column(
        Modifier.fillMaxSize().background(Ink.Bg).statusBarsPadding().navigationBarsPadding()
            .padding(start = 20.dp, end = 20.dp, top = 8.dp, bottom = 34.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Row(Modifier.fillMaxWidth().height(44.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Done", Modifier.clickable(onClick = onDone).padding(horizontal = 4.dp, vertical = 10.dp), style = Type.Button.copy(fontWeight = FontWeight.Normal))
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Icon(DesignIcons.Check, null, Modifier.size(16.dp), tint = Ink.Green)
                Text("Saved to Photos", style = Type.Row)
            }
            Box(Modifier.width(44.dp))
        }

        Box(Modifier.weight(1f).fillMaxWidth().clip(RoundedCornerShape(28.dp)).background(Color(0xFF111111))) {
            image?.let {
                Image(it.asImageBitmap(), "Captured photo with ${shot.shade.name} hair", Modifier.fillMaxSize(), contentScale = ContentScale.Crop, alignment = Alignment.TopCenter)
            }
            Row(
                Modifier.align(Alignment.BottomStart).padding(16.dp).clip(CircleShape).background(Color.Black.copy(alpha = 0.55f))
                    .padding(start = 6.dp, end = 12.dp, top = 6.dp, bottom = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Box(Modifier.size(20.dp).clip(CircleShape).background(Color(shot.shade.argb)))
                Text("${shot.shade.name} · ${shot.percent}%", style = Type.SmallBold)
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            PillButton("Share", { ctx.startActivity(CaptureSaver.shareIntent(shot.uri)) }, Modifier.fillMaxWidth(), light = true, icon = DesignIcons.Share)
            PillButton("Try another shade", onAnother, Modifier.fillMaxWidth())
        }
    }
}
