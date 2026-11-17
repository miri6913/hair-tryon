package com.vto.hairtryon.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type

/** Full-width pill. [light] = white pill, black label (primary); otherwise [dark] fill with white label. */
@Composable
fun PillButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    light: Boolean = false,
    fill: Color = Ink.Surface,
    height: Dp = 52.dp,
    icon: ImageVector? = null,
) {
    Row(
        modifier
            .height(height)
            .clip(CircleShape)
            .background(if (light) Ink.Text else fill)
            .clickable(role = Role.Button, onClick = onClick),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) {
            Icon(icon, null, Modifier.size(18.dp), tint = if (light) Ink.Bg else Ink.Text)
            Spacer(Modifier.width(8.dp))
        }
        Text(text, style = Type.Button.copy(color = if (light) Ink.Bg else Ink.Text))
    }
}

/** 44 dp round translucent button used for back / close / share / camera switch on top of a photo or the camera. */
@Composable
fun RoundIconButton(
    icon: ImageVector,
    description: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    bg: Color = Ink.Glass,
    size: Dp = 44.dp,
    tint: Color = Ink.Text,
    iconSize: Dp = 18.dp,
) {
    Box(
        modifier
            .size(size)
            .clip(CircleShape)
            .background(bg)
            .semantics { contentDescription = description }
            .clickable(role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { Icon(icon, null, Modifier.size(iconSize), tint = tint) }
}

/** Small frosted label ("Before", "Live · on device", ...). */
@Composable
fun GlassChip(modifier: Modifier = Modifier, bg: Color = Ink.GlassDark, content: @Composable () -> Unit) {
    Row(
        modifier.clip(CircleShape).background(bg).padding(horizontal = 10.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) { content() }
}

@Composable
fun GlassChip(text: String, modifier: Modifier = Modifier) = GlassChip(modifier) { Text(text, style = Type.Chip) }

/** COLOR / STYLE switch under the viewfinder: the current mode is bright, the other navigates. */
@Composable
fun ModeTabs(styleSelected: Boolean, onColor: () -> Unit, onStyle: () -> Unit, modifier: Modifier = Modifier) {
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(28.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("COLOR", Modifier.clickable(onClick = onColor).padding(vertical = 6.dp), style = Type.Tab.copy(color = if (styleSelected) Ink.Text4 else Ink.Text))
        Text("STYLE", Modifier.clickable(onClick = onStyle).padding(vertical = 6.dp), style = Type.Tab.copy(color = if (styleSelected) Ink.Text else Ink.Text4))
    }
}

/** The 76 dp shutter: white ring with a white disc. */
@Composable
fun Shutter(description: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier
            .size(76.dp)
            .border(4.dp, Ink.Text, CircleShape)
            .padding(8.dp)
            .clip(CircleShape)
            .background(Ink.Text)
            .semantics { contentDescription = description }
            .clickable(role = Role.Button, onClick = onClick),
    )
}

/** 48 dp rounded-square button with a 1 dp outline (photo library). */
@Composable
fun SquareIconButton(icon: ImageVector, description: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier
            .size(48.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(Ink.Surface)
            .border(1.dp, Ink.Hairline, RoundedCornerShape(14.dp))
            .semantics { contentDescription = description }
            .clickable(role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { Icon(icon, null, Modifier.size(22.dp), tint = Ink.Text) }
}

/** Three-slot row for the camera controls: left button, shutter, right button. */
@Composable
fun ControlRow(left: @Composable () -> Unit, center: @Composable () -> Unit, right: @Composable () -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 40.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) { left(); center(); right() }
}
