package com.vto.hairtryon.ui.hairstyle

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.vto.hairtryon.hairstyle.Styles
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type

/** Design board "Hairstyle · choose catalog": bottom sheet over Home to pick Women's, Men's or all styles. */
@Composable
fun CatalogSheet(onPick: (Catalog) -> Unit, onDismiss: () -> Unit) {
    Box(
        Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.62f))
            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onClick = onDismiss),
    ) {
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(8.dp).navigationBarsPadding()
                .clip(RoundedCornerShape(36.dp)).background(Ink.Surface)
                .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {} // swallow taps on the sheet itself
                .padding(start = 20.dp, end = 20.dp, top = 10.dp, bottom = 22.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            Box(Modifier.width(36.dp).height(5.dp).clip(RoundedCornerShape(3.dp)).background(Ink.Surface3))
            Column(Modifier.padding(top = 4.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Which styles would you like?", style = Type.Title, textAlign = TextAlign.Center)
                Text("You can switch any time", style = Type.Body15.copy(color = Ink.Text4), textAlign = TextAlign.Center)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                CatalogCard("Women's styles", "f_short_italian_bob", Modifier.weight(1f)) { onPick(Catalog.Women) }
                CatalogCard("Men's styles", "m_medium_quiff", Modifier.weight(1f)) { onPick(Catalog.Men) }
            }
            Text(
                "Show all styles",
                Modifier.heightIn(min = 44.dp).clickable(role = Role.Button) { onPick(Catalog.All) }.padding(horizontal = 12.dp, vertical = 12.dp),
                style = Type.Body15.copy(color = Ink.Text4),
            )
        }
    }
}

@Composable
private fun CatalogCard(label: String, sketchId: String, modifier: Modifier, onClick: () -> Unit) {
    val res = Styles.sketchRes(LocalContext.current, sketchId)
    Column(
        modifier.semantics { contentDescription = label }.clickable(role = Role.Button, onClick = onClick),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(Modifier.fillMaxWidth().height(190.dp).clip(RoundedCornerShape(20.dp)).background(CardOff)) {
            if (res != 0) {
                Image(
                    painterResource(res), null, Modifier.fillMaxSize().padding(top = 14.dp),
                    contentScale = ContentScale.Fit, alignment = Alignment.BottomCenter,
                    colorFilter = ColorFilter.tint(Ink.Surface),
                )
            }
        }
        Text(label, style = Type.Row, textAlign = TextAlign.Center)
    }
}
