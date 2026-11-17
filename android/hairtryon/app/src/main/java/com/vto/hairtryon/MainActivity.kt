package com.vto.hairtryon

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.core.content.ContextCompat
import com.vto.hairtryon.ui.App
import com.vto.hairtryon.ui.theme.HairTheme

class MainActivity : ComponentActivity() {
    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) showApp()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // black app under transparent bars with light icons; every screen pads itself with the system-bar insets
        enableEdgeToEdge(statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT), navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT))
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            showApp()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    // debug: `adb shell am start -n com.vto.hairtryon/.MainActivity --es still <path to jpg/png>` renders a photo instead of the camera
    private fun showApp() {
        val still = intent.getStringExtra("still")
        val refine = intent.getBooleanExtra("refine", true) // `--ez refine false` for the golden test
        val shade = intent.getStringExtra("shade") // palette key, e.g. `--es shade honey`
        val debug = intent.getBooleanExtra("debug", false) // `--ez debug true` shows Refine / Prior / Mask / Flip
        // any of these debug extras means a test run: open Live Color directly instead of Home
        val straightToLive = listOf("still", "refine", "shade", "debug").any { intent.hasExtra(it) }
        setContent {
            HairTheme {
                App(owner = this, startInLiveColor = straightToLive, stillPath = still, initialRefine = refine, initialShade = shade, showDebug = debug)
            }
        }
    }
}
