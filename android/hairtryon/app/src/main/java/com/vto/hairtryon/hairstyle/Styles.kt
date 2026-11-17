package com.vto.hairtryon.hairstyle

import android.content.Context

/** One catalog entry. Ids and groups match server/hairapi/catalog.json; the API ships no thumbnails, so the app has its own sketches. */
class HairStyle(val id: String, val name: String)

class StyleGroup(val label: String, val styles: List<HairStyle>)

object Styles {
    val female = listOf(
        StyleGroup("SHORT", listOf(HairStyle("f_short_pixie", "Pixie cut"), HairStyle("f_short_bob", "Short bob"), HairStyle("f_short_italian_bob", "Italian bob"))),
        StyleGroup("MEDIUM", listOf(HairStyle("f_medium_layered", "Layered shoulder"), HairStyle("f_medium_bob_bangs", "Bob with bangs"))),
        StyleGroup("LONG", listOf(HairStyle("f_long_straight", "Long straight"), HairStyle("f_long_side_ponytail", "Low side ponytail"), HairStyle("f_long_butterfly", "Butterfly cut"), HairStyle("f_long_volume", "Long volume"))),
    )
    val male = listOf(
        StyleGroup("SHORT", listOf(HairStyle("m_short_buzz", "Buzz cut"), HairStyle("m_short_textured_crop", "Textured crop"), HairStyle("m_short_comma", "Comma hair"))),
        StyleGroup("MEDIUM", listOf(HairStyle("m_medium_quiff", "Swept-back quiff"), HairStyle("m_medium_side_part", "Side part"))),
        StyleGroup("LONG", listOf(HairStyle("m_long_shoulder", "Shoulder-length"), HairStyle("m_long_man_bun", "Man bun"))),
    )

    const val DEFAULT_FEMALE = "f_medium_layered"
    const val DEFAULT_MALE = "m_medium_quiff"

    fun groups(female: Boolean) = if (female) Styles.female else male

    fun byId(id: String?): HairStyle? = (female + male).flatMap { it.styles }.firstOrNull { it.id == id }

    /** res/drawable-nodpi/sketch_<id>.png (alpha masks of the line art). 0 if missing. */
    fun sketchRes(ctx: Context, id: String): Int = ctx.resources.getIdentifier("sketch_$id", "drawable", ctx.packageName)
}
