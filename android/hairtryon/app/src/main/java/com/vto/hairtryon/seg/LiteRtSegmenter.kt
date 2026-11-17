package com.vto.hairtryon.seg

import android.content.Context
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.vto.hairtryon.palette.ColorMath
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * Hair segmentation with LiteRT CompiledModel on GPU (FP16), run on its own thread.
 * Input: RGBA8 512x512, top row first. Output: per-pixel logit difference (hair - background).
 *
 * The model, GPU context and buffers are created lazily on the worker thread and only ever used there.
 */
class LiteRtSegmenter(private val ctx: Context) {
    private val thread = HandlerThread("seg").also { it.start() }
    private val handler = Handler(thread.looper)
    private val busy = AtomicBoolean(false)

    private val inBytes = ByteArray(SIZE * SIZE * 4)
    private val inFloats = FloatArray(SIZE * SIZE * 4)
    private val diff = FloatArray(SIZE * SIZE)

    /** Written by the worker, read on the GL thread after [takeMask] returns it. */
    private val maskBuf: FloatBuffer = ByteBuffer.allocateDirect(SIZE * SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val maskReady = AtomicBoolean(false)

    private var model: CompiledModel? = null
    private var inputs: List<com.google.ai.edge.litert.TensorBuffer>? = null
    private var outputs: List<com.google.ai.edge.litert.TensorBuffer>? = null

    private var pendingTs = 0L

    /**
     * Feed the previous mask back as the model's 4th input channel. Off by default: scripts/prior_flicker_test.py
     * measured no change in edge flicker (-0.7%) under sensor noise.
     */
    @Volatile var usePrior = false

    /** Mean frame-to-frame change of the hair probability on boundary pixels (lower = steadier edge). */
    @Volatile var edgeFlicker = 0f

    private val lastP = FloatArray(SIZE * SIZE)
    private var statsCounter = 0

    private fun sigmoid(x: Float): Float {
        val t = ((x + SIGMOID_RANGE) * SIGMOID_SCALE).toInt().coerceIn(0, SIGMOID_LUT.size - 1)
        return SIGMOID_LUT[t]
    }

    /** Camera timestamp (ns) of the frame the latest finished mask was computed from. */
    @Volatile var maskTimestampNs = 0L

    /** Mean L* of pixels classified as hair in the latest mask; NaN until enough hair has been seen. */
    @Volatile var meanHairL = Float.NaN

    /**
     * Lightness of the person's hair at the 2/10/50/90/98th percentiles, measured on 4x4-pixel block means of
     * confident hair (about the Gaussian base the recolor works on, so noise does not widen the range). Null until
     * enough hair has been seen. [hairA] / [hairB]: mean chroma of the hair.
     */
    @Volatile var hairQuantiles: FloatArray? = null
    @Volatile var hairA = 0f
    @Volatile var hairB = 0f
    private val hist = IntArray(HIST_BINS)
    @Volatile var hairPixels = 0
    @Volatile var breakdown = ""
    @Volatile var lastMs = 0f
    @Volatile var fps = 0f
    private var winStart = 0L
    private var winCount = 0

    val isIdle: Boolean get() = !busy.get()

    /** Copies the 512x512 RGBA frame and starts inference. Ignored if the worker is still busy. */
    fun submit(rgba: ByteBuffer, frameTimestampNs: Long): Boolean {
        if (!busy.compareAndSet(false, true)) return false
        rgba.rewind()
        rgba.get(inBytes)
        pendingTs = frameTimestampNs
        handler.post { process() }
        return true
    }

    /** Latest mask if a new one finished since the last call (position reset for upload), else null. */
    fun takeMask(): FloatBuffer? {
        if (!maskReady.getAndSet(false)) return null
        maskBuf.rewind()
        return maskBuf
    }

    private fun process() {
        try {
            ensureModel()
            val t0 = SystemClock.elapsedRealtimeNanos()
            val b = inBytes
            val f = inFloats
            val prior = usePrior
            val lp = lastP
            for (i in 0 until SIZE * SIZE) {
                f[i * 4] = (b[i * 4].toInt() and 0xFF) / 255f
                f[i * 4 + 1] = (b[i * 4 + 1].toInt() and 0xFF) / 255f
                f[i * 4 + 2] = (b[i * 4 + 2].toInt() and 0xFF) / 255f
                // 4th channel = previous frame's hair probability: the model was built for video and uses it
                // to stay temporally consistent (MediaPipe feeds it the same way)
                f[i * 4 + 3] = if (prior) lp[i] else 0f
            }
            val t1 = SystemClock.elapsedRealtimeNanos()
            inputs!![0].writeFloat(f)
            val t2 = SystemClock.elapsedRealtimeNanos()
            model!!.run(inputs!!, outputs!!)
            val out = outputs!![0].readFloat()
            val t3 = SystemClock.elapsedRealtimeNanos()
            val d = diff
            val statsFrame = (statsCounter++ and 3) == 0 // hair lightness changes slowly: every 4th mask is plenty
            var lSum = 0.0
            var hairPx = 0
            var flickSum = 0.0
            var flickCount = 0
            for (i in 0 until SIZE * SIZE) {
                val v = out[i * 2 + 1] - out[i * 2]
                d[i] = v
                // sigmoid only where needed: every pixel when the prior is fed back, every 4th for the flicker meter
                if (prior || (MEASURE_FLICKER && (i and 3) == 0)) {
                    val p = sigmoid(v)
                    if (MEASURE_FLICKER && (i and 3) == 0) {
                        // edge flicker: mean |p_t - p_(t-1)| over pixels that are not clearly background or clearly hair
                        val pp = lp[i]
                        if ((p > 0.05f || pp > 0.05f) && (p < 0.95f || pp < 0.95f)) {
                            flickSum += abs(p - pp)
                            flickCount++
                        }
                    }
                    lp[i] = p
                }
                if (statsFrame && v > 0f) { // hair pixel: accumulate lightness for the recolor's mean base lightness
                    lSum += ColorMath.lightness(b[i * 4].toInt() and 0xFF, b[i * 4 + 1].toInt() and 0xFF, b[i * 4 + 2].toInt() and 0xFF)
                    hairPx++
                }
            }
            if (flickCount > 50) edgeFlicker = (flickSum / flickCount).toFloat()
            if (statsFrame) {
                hairPixels = hairPx
                if (hairPx >= MIN_HAIR_PIXELS) meanHairL = (lSum / hairPx).toFloat()
                measureHair(b, d)
            }
            maskBuf.rewind()
            maskBuf.put(d) // bulk copy; per-element puts into a direct buffer were ~20 ms
            maskTimestampNs = pendingTs
            maskReady.set(true)
            val t4 = SystemClock.elapsedRealtimeNanos()
            lastMs = (t4 - t0) / 1e6f
            breakdown = "pack=%.1f write=%.1f run+read=%.1f post=%.1f".format(
                (t1 - t0) / 1e6f, (t2 - t1) / 1e6f, (t3 - t2) / 1e6f, (t4 - t3) / 1e6f,
            )
            countFps()
        } catch (t: Throwable) {
            Log.e(TAG, "segmentation failed", t)
        } finally {
            busy.set(false)
        }
    }

    /** Hair lightness quantiles and mean chroma from the frame and its mask logits (worker thread, ~every 4th mask). */
    private fun measureHair(rgba: ByteArray, logit: FloatArray) {
        java.util.Arrays.fill(hist, 0)
        var blocks = 0
        var sr = 0L
        var sg = 0L
        var sb = 0L
        var n = 0
        var by = 0
        while (by < SIZE) {
            var bx = 0
            while (bx < SIZE) {
                var cnt = 0
                var lSum = 0f
                var r = 0
                var g = 0
                var bl = 0
                for (y in by until by + BLOCK) {
                    for (x in bx until bx + BLOCK) {
                        val i = y * SIZE + x
                        if (logit[i] > 0f) {
                            val pr = rgba[i * 4].toInt() and 0xFF
                            val pg = rgba[i * 4 + 1].toInt() and 0xFF
                            val pb = rgba[i * 4 + 2].toInt() and 0xFF
                            lSum += ColorMath.lightness(pr, pg, pb)
                            r += pr; g += pg; bl += pb
                            cnt++
                        }
                    }
                }
                if (cnt >= BLOCK * BLOCK * 3 / 4) { // interior blocks only: edges mix in background
                    val l = lSum / cnt
                    hist[(l * HIST_BINS / 100f).toInt().coerceIn(0, HIST_BINS - 1)]++
                    blocks++
                    sr += r; sg += g; sb += bl; n += cnt
                }
                bx += BLOCK
            }
            by += BLOCK
        }
        if (blocks < MIN_BLOCKS) return
        val q = FloatArray(5)
        val targets = doubleArrayOf(0.02, 0.10, 0.50, 0.90, 0.98)
        var cum = 0
        var k = 0
        for (bin in 0 until HIST_BINS) {
            cum += hist[bin]
            while (k < 5 && cum >= targets[k] * blocks) {
                q[k] = (bin + 0.5f) * 100f / HIST_BINS
                k++
            }
        }
        while (k < 5) q[k++] = 100f
        hairQuantiles = q
        val lab = ColorMath.rgbToLab((sr / n).toInt(), (sg / n).toInt(), (sb / n).toInt())
        hairA = lab[1]
        hairB = lab[2]
    }

    private fun ensureModel() {
        if (model != null) return
        val t0 = SystemClock.elapsedRealtime()
        val env = Environment.create()
        val m = CompiledModel.create(
            ctx.assets, MODEL_FILE,
            CompiledModel.Options(Accelerator.GPU).apply {
                gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16)
            },
            env,
        )
        model = m
        inputs = m.createInputBuffers()
        outputs = m.createOutputBuffers()
        Log.i(TAG, "model ready in ${SystemClock.elapsedRealtime() - t0} ms")
    }

    private fun countFps() {
        val now = SystemClock.elapsedRealtime()
        if (winStart == 0L) winStart = now
        winCount++
        if (now - winStart >= 1000) {
            fps = winCount * 1000f / (now - winStart)
            Log.i("VTO_FPS", "seg_fps=%.1f seg_ms=%.1f [%s]".format(fps, lastMs, breakdown))
            winStart = now
            winCount = 0
        }
    }

    fun close() {
        handler.post {
            model?.close()
            model = null
            thread.quitSafely()
        }
    }

    companion object {
        const val SIZE = 512
        private const val MIN_HAIR_PIXELS = 300
        private const val BLOCK = 4
        private const val MIN_BLOCKS = 60
        private const val HIST_BINS = 200

        /** Debug meter for scripts/prior_flicker_test.py-style checks on device; costs a few ms per mask. */
        private const val MEASURE_FLICKER = false
        private const val SIGMOID_RANGE = 16f
        private const val SIGMOID_SCALE = 32f
        private val SIGMOID_LUT = FloatArray((2 * SIGMOID_RANGE * SIGMOID_SCALE).toInt() + 1) {
            val x = it / SIGMOID_SCALE - SIGMOID_RANGE
            (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
        }
        private const val MODEL_FILE = "hair_segmenter.tflite"
        private const val TAG = "VTO_SEG"
    }
}
