package com.vto.hairtryon.camera

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.vto.hairtryon.gl.CameraRenderer

/** Binds a CameraX Preview whose surface is the renderer's SurfaceTexture. */
class CameraSource(private val ctx: Context) {
    private var closed = false

    fun bind(owner: LifecycleOwner, front: Boolean, renderer: CameraRenderer) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            if (closed) return@addListener // the screen left while the provider was still starting
            val provider = future.get()
            val preview = Preview.Builder().build()
            preview.setSurfaceProvider(ContextCompat.getMainExecutor(ctx)) { request -> renderer.provideSurface(request) }
            val selector = if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            try {
                provider.unbindAll()
                provider.bindToLifecycle(owner, selector, preview)
            } catch (e: Exception) {
                Log.e("VTO_CAMERA", "bind failed", e)
            }
        }, ContextCompat.getMainExecutor(ctx))
    }

    /** Releases the camera when the Live Color screen goes away (the next screen binds its own use cases). */
    fun unbind() {
        closed = true
        try {
            ProcessCameraProvider.getInstance(ctx).get().unbindAll()
        } catch (e: Exception) {
            Log.w("VTO_CAMERA", "unbind failed", e)
        }
    }
}
