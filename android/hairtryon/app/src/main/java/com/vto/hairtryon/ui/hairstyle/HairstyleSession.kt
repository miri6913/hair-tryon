package com.vto.hairtryon.ui.hairstyle

import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.vto.hairtryon.hairstyle.HairApi
import com.vto.hairtryon.hairstyle.Styles

/** Which styles the user asked for on the catalog sheet. All shows the Women | Men tabs. */
enum class Catalog { Women, Men, All }

/** State of one hairstyle try, shared by Take photo -> Choose style -> Generating -> Result. Lives at the app root. */
class HairstyleSession {
    var photo by mutableStateOf<Bitmap?>(null) // upright, as the user saw it (front camera already mirrored)
    var catalog by mutableStateOf(Catalog.All)
    var female by mutableStateOf(true)
    var styleId by mutableStateOf(Styles.DEFAULT_FEMALE)
    var result by mutableStateOf<Bitmap?>(null)
    var seconds by mutableStateOf(0f)
    var error by mutableStateOf<HairApi.Failure?>(null)

    fun selectGender(female: Boolean) {
        this.female = female
        styleId = if (female) Styles.DEFAULT_FEMALE else Styles.DEFAULT_MALE
    }

    fun chooseCatalog(c: Catalog) {
        catalog = c
        when (c) {
            Catalog.Women -> selectGender(true)
            Catalog.Men -> selectGender(false)
            Catalog.All -> {}
        }
    }

    fun newPhoto(bitmap: Bitmap) {
        photo = bitmap
        result = null
        error = null
    }
}
