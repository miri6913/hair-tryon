package com.vto.hairtryon.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.vto.hairtryon.R

/** Tokens from the design canvas (data/ui_design/final_ui): black ground, Apple-style greys, pill buttons. */
object Ink {
    val Bg = Color(0xFF000000)
    val Text = Color(0xFFF5F5F7)
    val Text2 = Color(0xFFC7C7CC)
    val Text3 = Color(0xFFAEAEB2)
    val Text4 = Color(0xFF98989D)
    val Surface = Color(0xFF1C1C1E)
    val Surface2 = Color(0xFF2C2C2E)
    val Surface3 = Color(0xFF48484A)
    val Glass = Color(0x8C1C1C1E) // rgba(28,28,30,.55): the design blurs behind it, here it is just translucent
    val GlassDark = Color(0x73000000)
    val Green = Color(0xFF30D158)
    val Hairline = Color(0x38FFFFFF)
}

val Inter = FontFamily(
    Font(R.font.inter_regular, FontWeight.Normal),
    Font(R.font.inter_medium, FontWeight.Medium),
    Font(R.font.inter_semibold, FontWeight.SemiBold),
    Font(R.font.inter_bold, FontWeight.Bold),
)

private fun t(size: Int, weight: FontWeight, line: Int? = null, spacing: Float = 0f) =
    TextStyle(fontFamily = Inter, fontSize = size.sp, fontWeight = weight, lineHeight = (line ?: (size * 1.3f).toInt()).sp, letterSpacing = spacing.sp, color = Ink.Text)

object Type {
    val Hero = t(40, FontWeight.Bold, 44, -1.2f)
    val Title = t(22, FontWeight.SemiBold, 28, -0.2f)
    val Bar = t(17, FontWeight.SemiBold)
    val Button = t(17, FontWeight.SemiBold)
    val Body = t(16, FontWeight.Normal, 22)
    val Body15 = t(15, FontWeight.Normal, 21)
    val Row = t(15, FontWeight.SemiBold)
    val Small = t(13, FontWeight.Normal)
    val SmallBold = t(13, FontWeight.SemiBold)
    val Chip = t(12, FontWeight.SemiBold)
    val Tab = t(13, FontWeight.SemiBold, spacing = 0.8f)
    val Caption = t(14, FontWeight.Medium)
}

@Composable
fun HairTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = Ink.Bg, surface = Ink.Bg, onBackground = Ink.Text, onSurface = Ink.Text,
            primary = Ink.Text, onPrimary = Ink.Bg,
        ),
        typography = Typography(
            bodyLarge = Type.Body, bodyMedium = Type.Body15, labelMedium = Type.SmallBold, labelSmall = Type.Chip,
        ),
        content = content,
    )
}
