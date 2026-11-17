package com.vto.hairtryon.ui

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.LifecycleOwner
import com.vto.hairtryon.hairstyle.HairApi.Failure
import com.vto.hairtryon.ui.hairstyle.CatalogSheet
import com.vto.hairtryon.ui.hairstyle.ErrorSheet
import com.vto.hairtryon.ui.hairstyle.GeneratingScreen
import com.vto.hairtryon.ui.hairstyle.HairstyleSession
import com.vto.hairtryon.ui.hairstyle.StyleCaptureScreen
import com.vto.hairtryon.ui.hairstyle.StylePickerScreen
import com.vto.hairtryon.ui.hairstyle.StyleResultScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private enum class Screen { Home, LiveColor, Captured, StyleCapture, StylePicker, Generating, StyleResult }

/** App root: a tiny state machine over the design's screens (no navigation library; there are only 7 and no deep links). */
@Composable
fun App(
    owner: LifecycleOwner,
    startInLiveColor: Boolean,
    stillPath: String?,
    initialRefine: Boolean,
    initialShade: String?,
    showDebug: Boolean,
) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var screen by remember { mutableStateOf(if (startInLiveColor) Screen.LiveColor else Screen.Home) }
    var shot by remember { mutableStateOf<CapturedShot?>(null) }
    val session = remember { HairstyleSession() }
    var showCatalog by remember { mutableStateOf(false) }

    val pickPhoto = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) scope.launch {
            val bmp = withContext(Dispatchers.IO) { PhotoLoader.load(ctx, uri, 2048) }
            if (bmp != null) {
                session.newPhoto(bmp)
                screen = Screen.StylePicker
            }
        }
    }
    fun library() = pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))

    BackHandler(enabled = screen != Screen.Home || showCatalog) {
        if (showCatalog) { showCatalog = false; return@BackHandler }
        screen = when (screen) {
            Screen.Captured -> Screen.LiveColor
            Screen.StylePicker -> Screen.StyleCapture
            Screen.Generating, Screen.StyleResult -> Screen.StylePicker
            else -> Screen.Home
        }
    }

    Box(Modifier.fillMaxSize()) {
        when (screen) {
            Screen.Home -> {
                HomeScreen(onLiveColor = { screen = Screen.LiveColor }, onHairstyle = { showCatalog = true })
                if (showCatalog) {
                    CatalogSheet(
                        onPick = { session.chooseCatalog(it); showCatalog = false; screen = Screen.StyleCapture },
                        onDismiss = { showCatalog = false },
                    )
                }
            }

            Screen.LiveColor -> TryOnScreen(
                owner = owner, onBack = { screen = Screen.Home }, onStyle = { screen = Screen.StyleCapture },
                onCaptured = { shot = it; screen = Screen.Captured },
                stillPath = stillPath, initialRefine = initialRefine, initialShade = initialShade, showDebug = showDebug,
            )

            Screen.Captured -> shot?.let { CapturedScreen(it, onDone = { screen = Screen.Home }, onAnother = { screen = Screen.LiveColor }) }

            Screen.StyleCapture -> StyleCaptureScreen(
                owner = owner, onClose = { screen = Screen.Home }, onColor = { screen = Screen.LiveColor }, onLibrary = ::library,
                onPhoto = { session.newPhoto(it); screen = Screen.StylePicker },
            )

            Screen.StylePicker -> {
                StylePickerScreen(session, onBack = { screen = Screen.StyleCapture }, onTry = { screen = Screen.Generating })
                session.error?.let { kind ->
                    ErrorSheet(
                        kind,
                        onPrimary = {
                            session.error = null
                            screen = when (kind) {
                                Failure.Busy -> Screen.Generating
                                Failure.Offline -> Screen.LiveColor
                                else -> Screen.StyleCapture
                            }
                        },
                        onSecondary = {
                            session.error = null
                            when (kind) {
                                Failure.Busy -> {} // stay on the styles
                                Failure.Offline -> screen = Screen.Home
                                else -> library()
                            }
                        },
                    )
                }
            }

            Screen.Generating -> GeneratingScreen(
                session, onCancel = { screen = Screen.StylePicker },
                onDone = { screen = Screen.StyleResult }, onFailed = { screen = Screen.StylePicker },
            )

            Screen.StyleResult -> StyleResultScreen(session, onClose = { screen = Screen.StylePicker }, onRetake = { screen = Screen.StyleCapture })
        }
    }
}
