package com.vto.hairtryon.ui.hairstyle

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.vto.hairtryon.hairstyle.HairStyle
import com.vto.hairtryon.hairstyle.Styles
import com.vto.hairtryon.ui.components.DesignIcons
import com.vto.hairtryon.ui.components.PillButton
import com.vto.hairtryon.ui.components.RoundIconButton
import com.vto.hairtryon.ui.theme.Ink
import com.vto.hairtryon.ui.theme.Type

internal val CardOff =Brush.verticalGradient(listOf(Color(0xFFE5E5EA), Color(0xFFAEAEB2)))
private val CardOn = Brush.verticalGradient(listOf(Color(0xFFF2F2F7), Color(0xFFC7C7CC)))

/** Hairstyle step 2: the catalog styles as sketch cards. Women / Men tabs only when the user chose "all styles"; otherwise just that gender. */
@Composable
fun StylePickerScreen(session: HairstyleSession, onBack: () -> Unit, onTry: () -> Unit) {
    val selected = Styles.byId(session.styleId)
    Box(Modifier.fillMaxSize().background(Ink.Bg)) {
        Column(Modifier.fillMaxSize().statusBarsPadding()) {
            Row(Modifier.fillMaxWidth().padding(top = 8.dp, start = 16.dp, end = 16.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
                RoundIconButton(DesignIcons.Back, "Back", onBack, bg = Ink.Surface)
                Text("Choose a style", style = Type.Bar)
                Box(Modifier.width(44.dp))
            }

            Column(Modifier.padding(start = 20.dp, end = 20.dp, top = 16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    session.photo?.let {
                        Image(it.asImageBitmap(), "Your photo", Modifier.size(52.dp).clip(RoundedCornerShape(14.dp)), contentScale = ContentScale.Crop, alignment = Alignment.TopCenter)
                    }
                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text("Your photo", style = Type.Row)
                        Text("Your own hair colour is kept", style = Type.Small.copy(color = Ink.Text4))
                    }
                    Text("Retake", Modifier.clickable(onClick = onBack).padding(horizontal = 2.dp, vertical = 10.dp), style = Type.Body15)
                }
                if (session.catalog == Catalog.All) GenderTabs(session.female) { session.selectGender(it) }
            }

            Column(
                Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(start = 20.dp, end = 20.dp, top = 20.dp, bottom = 24.dp),
                verticalArrangement = Arrangement.spacedBy(22.dp),
            ) {
                for (group in Styles.groups(session.female)) {
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(group.label, style = Type.Tab.copy(color = Ink.Text4))
                        for (pair in group.styles.chunked(2)) {
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                for (style in pair) {
                                    StyleCard(style, style.id == session.styleId, Modifier.weight(1f)) { session.styleId = style.id }
                                }
                                if (pair.size == 1) Spacer(Modifier.weight(1f))
                            }
                        }
                    }
                }
            }

            Column(
                Modifier.fillMaxWidth().background(Color.Black.copy(alpha = 0.92f)).navigationBarsPadding()
                    .padding(start = 20.dp, end = 20.dp, top = 14.dp, bottom = 34.dp),
            ) {
                PillButton("Try ${selected?.name ?: "style"}", onTry, Modifier.fillMaxWidth(), light = true)
            }
        }
    }
}

@Composable
private fun GenderTabs(female: Boolean, onSelect: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Ink.Surface).padding(2.dp), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
        for ((label, isFemale) in listOf("Women" to true, "Men" to false)) {
            val on = female == isFemale
            Box(
                Modifier.weight(1f).height(34.dp).clip(RoundedCornerShape(8.dp)).background(if (on) Ink.Surface3 else Color.Transparent)
                    .semantics { role = Role.Tab; this.selected = on }
                    .clickable { onSelect(isFemale) },
                contentAlignment = Alignment.Center,
            ) { Text(label, style = Type.Row.copy(fontSize = 14.sp)) }
        }
    }
}

/** A 150 dp silver card with the style's line sketch in ink; the selected card is brighter and ringed. */
@Composable
private fun StyleCard(style: HairStyle, on: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val ctx = LocalContext.current
    val res = Styles.sketchRes(ctx, style.id)
    Column(
        modifier.semantics { contentDescription = style.name; selected = on }.clickable(role = Role.RadioButton, onClick = onClick),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(Modifier.then(if (on) Modifier.border(2.dp, Ink.Text, RoundedCornerShape(22.dp)) else Modifier).padding(4.dp)) {
            Box(Modifier.fillMaxWidth().height(150.dp).clip(RoundedCornerShape(18.dp)).background(if (on) CardOn else CardOff)) {
                if (res != 0) {
                    Image(
                        painterResource(res), null, Modifier.fillMaxSize().padding(top = 10.dp),
                        contentScale = ContentScale.Fit, alignment = Alignment.BottomCenter,
                        colorFilter = ColorFilter.tint(Ink.Surface), // the PNGs are white line art in the alpha channel
                    )
                }
            }
        }
        Text(style.name, Modifier.padding(start = 6.dp), style = Type.Caption)
    }
}
