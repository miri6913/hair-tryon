package com.vto.hairtryon.ui.components

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.unit.dp

/** The design's 24x24 stroke icons (path data copied from the canvas SVGs). Tint them through Icon(tint = ...). */
object DesignIcons {
    private fun icon(name: String, width: Float = 2f, vararg paths: String): ImageVector =
        ImageVector.Builder(name, 24.dp, 24.dp, 24f, 24f).apply {
            for (p in paths) {
                addPath(
                    pathData = addPathNodes(p), stroke = SolidColor(Color.White), strokeLineWidth = width,
                    strokeLineCap = StrokeCap.Round, strokeLineJoin = StrokeJoin.Round,
                )
            }
        }.build()

    private const val RECT_PHOTO = "M6 4h12a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3z"

    val Back = icon("back", 2.4f, "M15 6l-6 6 6 6")
    val Close = icon("close", 2.4f, "M6 6l12 12M18 6L6 18")
    val Check = icon("check", 2.6f, "M5 12.5l4.5 4.5L19 7.5")
    val Share = icon("share", 2.2f, "M12 15V3M7.5 7.5L12 3l4.5 4.5", "M5 12v6.5A2.5 2.5 0 0 0 7.5 21h9a2.5 2.5 0 0 0 2.5-2.5V12")
    val Gallery = icon("gallery", 2f, RECT_PHOTO, "M9 8a2 2 0 1 0 0 4 2 2 0 0 0 0-4z", "M21 16l-5-5-9 9")
    val Compare = icon("compare", 2f, RECT_PHOTO, "M12 2v20")
    val Drop = icon("drop", 2f, "M12 3c3.5 4 6 7.2 6 10.5A6 6 0 0 1 6 13.5C6 10.2 8.5 7 12 3z")
    val Handle = icon("handle", 2.4f, "M9 7l-5 5 5 5M15 7l5 5-5 5")
    val Lock = icon("lock", 2f, "M7.5 11h9a2.5 2.5 0 0 1 2.5 2.5v5a2.5 2.5 0 0 1-2.5 2.5h-9A2.5 2.5 0 0 1 5 18.5v-5A2.5 2.5 0 0 1 7.5 11z", "M8 11V8a4 4 0 0 1 8 0v3")
    val SwitchCamera = icon(
        "switch_camera", 2f,
        "M3 8.5A2.5 2.5 0 0 1 5.5 6H7l1.5-2h7L17 6h1.5A2.5 2.5 0 0 1 21 8.5v8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5z",
        "M9 12.5a3 3 0 0 1 5.2-2M15 11.5a3 3 0 0 1-5.2 2", "M14.5 8.5v2h-2M9.5 15.5v-2h2",
    )

    // error sheet glyphs (stroke 1.8)
    val NoFace = icon("no_face", 1.8f, "M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M9.5 10v.5M14.5 10v.5M9.5 15.5c1.4 1 3.6 1 5 0")
    val People = icon("people", 1.8f, "M9 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM2.5 20c.6-3.4 3.2-5.5 6.5-5.5s5.9 2.1 6.5 5.5M16 4.3a3.5 3.5 0 0 1 0 6.4M18 14.8c1.9.8 3.1 2.6 3.5 5.2")
    val Clock = icon("clock", 1.8f, "M12 7v5l3 2M21 12a9 9 0 1 1-9-9 9 9 0 0 1 9 9z")
    val Offline = icon("offline", 1.8f, "M3 3l18 18M8.5 8.6A5 5 0 0 0 6 17.5h10.5M17.7 17.3A4 4 0 0 0 17 9.5a6 6 0 0 0-6.6-4.4")
}
