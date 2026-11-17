package com.vto.hairtryon.hairstyle

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.vto.hairtryon.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume

/** Client for server/hairapi (`POST /v1/hairstyle`). The server keeps the photo's aspect ratio, so the result lines up with the upload. */
object HairApi {
    enum class Failure { NoFace, MultipleFaces, BadImage, Busy, Offline }

    sealed interface Outcome {
        class Ok(val image: Bitmap, val seconds: Float, val headCutOff: Boolean) : Outcome
        class Fail(val kind: Failure) : Outcome
    }

    private const val MAX_SIDE = 1024

    // read timeout covers the server queue (MAX_QUEUE 12 x ~4.5 s) and stays under Cloudflare's ~100 s request limit
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(100, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    /** Scales [photo] to a 1024 px long side and encodes it as JPEG q90 (the server downsizes to this anyway). */
    private fun encode(photo: Bitmap): ByteArray {
        val s = MAX_SIDE.toFloat() / maxOf(photo.width, photo.height)
        val bmp = if (s < 1f) Bitmap.createScaledBitmap(photo, (photo.width * s).toInt(), (photo.height * s).toInt(), true) else photo
        return ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) }.toByteArray()
    }

    /** Restyles [photo]. Cancelling the calling coroutine cancels the HTTP call. */
    suspend fun restyle(photo: Bitmap, styleId: String): Outcome {
        if (BuildConfig.HAIRAPI_KEY.isEmpty()) return Outcome.Fail(Failure.Offline) // no key configured: nothing to call
        val jpeg = withContext(Dispatchers.Default) { encode(photo) }
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("style_id", styleId)
            .addFormDataPart("photo", "photo.jpg", jpeg.toRequestBody("image/jpeg".toMediaType()))
            .build()
        val request = Request.Builder()
            .url(BuildConfig.HAIRAPI_URL.trimEnd('/') + "/v1/hairstyle")
            .header("X-API-Key", BuildConfig.HAIRAPI_KEY)
            .post(body)
            .build()
        val t0 = System.nanoTime()
        val response = try {
            call(client.newCall(request))
        } catch (e: IOException) {
            return Outcome.Fail(Failure.Offline)
        }
        return withContext(Dispatchers.Default) {
            response.use { r ->
                if (r.isSuccessful) {
                    val bytes = r.body?.bytes() ?: return@use Outcome.Fail(Failure.Offline)
                    val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return@use Outcome.Fail(Failure.Offline)
                    val seconds = (System.nanoTime() - t0) / 1e9f
                    Outcome.Ok(bmp, seconds, r.header("X-Warnings")?.contains("head_cut_off") == true)
                } else {
                    Outcome.Fail(failureOf(r.code, r.body?.string()))
                }
            }
        }
    }

    // photo problems come as {"detail": {"code", "message"}}; everything else has a string detail (see server/hairapi/README.md)
    private fun failureOf(status: Int, body: String?): Failure {
        val code = try {
            JSONObject(body ?: "").optJSONObject("detail")?.optString("code")
        } catch (e: Exception) {
            null
        }
        return when {
            code == "no_face" -> Failure.NoFace
            code == "multiple_faces" -> Failure.MultipleFaces
            code == "bad_image" || status == 413 -> Failure.BadImage
            status == 429 || status == 503 -> Failure.Busy
            else -> Failure.Offline // 401 (wrong key), 5xx, unknown style id: not something the user can fix by retaking
        }
    }

    private suspend fun call(call: Call): Response = suspendCancellableCoroutine { cont ->
        cont.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (!cont.isCancelled) cont.resumeWith(Result.failure(e))
            }

            override fun onResponse(call: Call, response: Response) = cont.resume(response)
        })
    }
}
