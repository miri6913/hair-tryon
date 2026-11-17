package com.vto.hairtryon.capture

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.core.content.FileProvider
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/** Writes captures to Pictures/HairTryOn (MediaStore on API 29+, an app-private file + FileProvider before) and shares them. */
object CaptureSaver {
    private const val TAG = "VTO_CAPTURE"
    private const val FOLDER = "HairTryOn"
    private val io = Executors.newSingleThreadExecutor()

    /** Saves [bitmap] as JPEG off the calling thread; [onDone] gets the shareable Uri, or null on failure. */
    fun saveJpeg(ctx: Context, bitmap: Bitmap, onDone: (Uri?) -> Unit) {
        io.execute {
            val uri = try {
                val name = "HairTryOn_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) saveMediaStore(ctx, bitmap, name) else savePrivate(ctx, bitmap, name)
            } catch (e: Exception) {
                Log.e(TAG, "save failed", e)
                null
            }
            onDone(uri)
        }
    }

    /**
     * Still-image debug mode: lossless PNG to `<external files>/still_out.png` so a golden comparison is not
     * polluted by JPEG noise, plus the smoothed mask the shader used as `still_mask.f32` (little-endian float32,
     * SIZE x SIZE, top row first). Pull both with `adb pull`; scripts/compare_golden.py consumes them.
     */
    fun saveStill(ctx: Context, bitmap: Bitmap, mask: FloatArray, onDone: (File?) -> Unit) {
        io.execute {
            val file = try {
                val dir = ctx.getExternalFilesDir(null)
                File(dir, "still_mask.f32").outputStream().use { out ->
                    val bb = java.nio.ByteBuffer.allocate(mask.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    bb.asFloatBuffer().put(mask)
                    out.write(bb.array())
                }
                File(dir, "still_out.png").also { f ->
                    f.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                }
            } catch (e: Exception) {
                Log.e(TAG, "png save failed", e)
                null
            }
            onDone(file)
        }
    }

    private fun saveMediaStore(ctx: Context, bitmap: Bitmap, name: String): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$FOLDER")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = ctx.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        try {
            resolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                ?: throw IllegalStateException("no output stream")
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw e
        }
        resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
        return uri
    }

    private fun savePrivate(ctx: Context, bitmap: Bitmap, name: String): Uri {
        val dir = File(ctx.getExternalFilesDir(Environment.DIRECTORY_PICTURES), FOLDER).apply { mkdirs() }
        val file = File(dir, name)
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        return FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", file)
    }

    fun shareIntent(uri: Uri): Intent = Intent.createChooser(
        Intent(Intent.ACTION_SEND).apply {
            type = "image/jpeg"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        },
        null,
    )
}
