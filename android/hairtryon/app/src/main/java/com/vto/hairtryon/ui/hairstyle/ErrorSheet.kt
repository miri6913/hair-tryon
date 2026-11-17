package com.vto.hairtryon.ui.hairstyle

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.vto.hairtryon.hairstyle.HairApi.Failure
import com.vto.hairtryon.ui.components.DesignIcons
import com.vto.hairtryon.ui.components.PillButton
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type

private class Copy(val icon: ImageVector, val title: String, val body: String, val primary: String, val secondary: String)

// copy from the design canvas (board "Errors"); BadImage has no board, it reuses the no-face wording
private fun copyOf(kind: Failure) = when (kind) {
    Failure.NoFace -> Copy(DesignIcons.NoFace, "We couldn’t find a face", "Face the camera in good light, with your whole head in the frame.", "Retake photo", "Choose from library")
    Failure.MultipleFaces -> Copy(DesignIcons.People, "More than one person", "Hairstyle works on one face at a time. Take a photo with just you in it.", "Retake photo", "Choose another photo")
    Failure.BadImage -> Copy(DesignIcons.NoFace, "We couldn’t read that photo", "Try a different photo, or take a new one.", "Retake photo", "Choose another photo")
    Failure.Busy -> Copy(DesignIcons.Clock, "Lots of people are trying styles", "Your photo wasn’t sent. Try again in about 10 seconds.", "Try again", "Back to styles")
    Failure.Offline -> Copy(DesignIcons.Offline, "Hairstyle is offline", "The styling server isn’t running right now. Live Color still works on your phone.", "Open Live Color", "Close")
}

/** Bottom sheet over a dimmed screen. [onPrimary] / [onSecondary] are the two buttons in the order of the design. */
@Composable
fun ErrorSheet(kind: Failure, onPrimary: () -> Unit, onSecondary: () -> Unit) {
    val c = copyOf(kind)
    Box(
        Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.55f))
            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {}, // modal: swallow taps on the dimmed screen
    ) {
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(8.dp).navigationBarsPadding()
                .clip(RoundedCornerShape(36.dp)).background(Ink.Surface)
                .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {} // swallow taps on the sheet itself
                .padding(start = 20.dp, end = 20.dp, top = 10.dp, bottom = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            Box(Modifier.width(36.dp).height(5.dp).clip(RoundedCornerShape(3.dp)).background(Ink.Surface3))
            Box(Modifier.padding(top = 8.dp).size(64.dp).clip(CircleShape).background(Ink.Surface2), contentAlignment = Alignment.Center) {
                Icon(c.icon, null, Modifier.size(30.dp), tint = Ink.Text)
            }
            Column(Modifier.padding(horizontal = 12.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(c.title, style = Type.Title, textAlign = TextAlign.Center)
                Text(c.body, style = Type.Body15.copy(color = Ink.Text3), textAlign = TextAlign.Center)
            }
            Column(Modifier.fillMaxWidth().padding(top = 4.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                PillButton(c.primary, onPrimary, Modifier.fillMaxWidth(), light = true)
                PillButton(c.secondary, onSecondary, Modifier.fillMaxWidth(), fill = Ink.Surface2)
            }
        }
    }
}
