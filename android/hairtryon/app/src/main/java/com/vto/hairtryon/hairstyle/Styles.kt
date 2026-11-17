package com.vto.hairtryon.hairstyle

import android.content.Context

/** One catalog entry. Ids and groups match server/hairapi/catalog.json; the API ships no thumbnails, so the app has its own sketches. */
class HairStyle(val id: String, val name: String)

class StyleGroup(val label: String, val styles: List<HairStyle>)

object Styles {
    val female = listOf(
        StyleGroup("SHORT", listOf(HairStyle("f_short_french_bob", "French bob"), HairStyle("f_short_bixie", "Bixie cut"), HairStyle("f_short_italian_bob", "Italian bob"))),
        StyleGroup("MEDIUM", listOf(HairStyle("f_medium_lob", "Collarbone lob"), HairStyle("f_medium_layered", "Layered shoulder"), HairStyle("f_medium_shag", "Shag cut"), HairStyle("f_medium_c_curl", "C-curl layers"))),
        StyleGroup("LONG", listOf(HairStyle("f_long_butterfly", "Butterfly cut"), HairStyle("f_long_curtain_layers", "Curtain layers"), HairStyle("f_long_volume", "Voluminous waves"))),
    )
    val male = listOf(
        StyleGroup("SHORT", listOf(HairStyle("m_short_buzz", "Buzz cut"), HairStyle("m_short_low_fade", "Low fade"), HairStyle("m_short_french_crop", "French crop"), HairStyle("m_short_ivy_league", "Ivy League"))),
        StyleGroup("MEDIUM", listOf(HairStyle("m_medium_two_block", "Two-block"), HairStyle("m_medium_modern_quiff", "Modern quiff"), HairStyle("m_medium_shadow_perm", "Shadow perm"), HairStyle("m_medium_leaf_cut", "Leaf cut"), HairStyle("m_medium_curtains", "Curtains"), HairStyle("m_medium_mullet", "Mullet"))),
    )

    const val DEFAULT_FEMALE = "f_medium_layered"
    const val DEFAULT_MALE = "m_medium_modern_quiff"

    fun groups(female: Boolean) = if (female) Styles.female else male

    fun byId(id: String?): HairStyle? = (female + male).flatMap { it.styles }.firstOrNull { it.id == id }

    /** res/drawable-nodpi/sketch_<id>.png (alpha masks of the line art). 0 if missing. */
    fun sketchRes(ctx: Context, id: String): Int = ctx.resources.getIdentifier("sketch_$id", "drawable", ctx.packageName)
}
