package com.vto.litertbench

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import kotlin.math.abs

object Bench {
    private class Spec(val file: String, val size: Int, val channels: Int)

    private val specs = mapOf(
        "hair" to Spec("hair_segmenter.tflite", 512, 4),            // RGB + previous-mask channel
        "multi" to Spec("selfie_multiclass_256x256.tflite", 256, 3),
    )

    private const val WARMUP = 10
    private const val ITERS = 100
    private const val SUSTAIN_SEC = 20

    private fun argmax(a: FloatArray, off: Int, n: Int): Int {
        var best = 0
        for (i in 1 until n) if (a[off + i] > a[off + best]) best = i
        return best
    }

    fun run(ctx: Context, modelKey: String, config: String, log: (String) -> Unit) {
        val spec = specs[modelKey] ?: return log("unknown model $modelKey")
        val tag = "model=$modelKey config=$config"
        try {
            val input = loadInput(ctx, spec)
            val cpuOut = if (config != "cpu") runOnce(ctx, spec, "cpu", input) else null
            val result = bench(ctx, spec, config, input, log, tag)

            var diff = ""
            if (cpuOut != null) {
                var maxD = 0f
                var sumD = 0.0
                for (i in cpuOut.indices) {
                    val d = abs(cpuOut[i] - result.output[i])
                    if (d > maxD) maxD = d
                    sumD += d
                }
                val classes = cpuOut.size / (spec.size * spec.size)
                var same = 0
                var cpuHair = 0
                var gpuHair = 0
                var both = 0
                var either = 0
                for (p in 0 until spec.size * spec.size) {
                    val a = argmax(cpuOut, p * classes, classes)
                    val b = argmax(result.output, p * classes, classes)
                    if (a == b) same++
                    if (a == 1) cpuHair++
                    if (b == 1) gpuHair++
                    if (a == 1 && b == 1) both++
                    if (a == 1 || b == 1) either++
                }
                diff = " vs_cpu_max_abs=%.4f vs_cpu_mean_abs=%.5f class_agree=%.4f hair_iou=%.4f hair_px_cpu=%d hair_px_gpu=%d"
                    .format(maxD, sumD / cpuOut.size, same.toDouble() / (spec.size * spec.size),
                        if (either == 0) 1.0 else both.toDouble() / either, cpuHair, gpuHair)
            }
            log(
                "$tag RESULT init_ms=%.0f run_median_ms=%.1f run_p95_ms=%.1f run_mean_ms=%.1f e2e_median_ms=%.1f out_len=%d%s"
                    .format(result.initMs, result.runMedian, result.runP95, result.runMean, result.e2eMedian, result.output.size, diff)
            )
        } catch (t: Throwable) {
            log("$tag FAILED ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    /** Spike (on-device FLUX blocks): time any .tflite given by absolute path, all inputs random, any number of inputs. */
    /**
     * Spike 3: simulate the denoising loop on the phone. Loads the single- and double-stream block graphs once and runs
     * `steps x (20 single + 5 double)` back to back, `reps` times in a row (heat throttling shows as rising rep times).
     * Every layer reuses the same weights, so the memory-bandwidth cost of 25 different layers is NOT included (optimistic).
     */
    fun runSim(singlePath: String, doublePath: String, steps: Int, reps: Int, config: String, log: (String) -> Unit) {
        val tag = "sim steps=$steps config=$config"
        try {
            val env = Environment.create()
            val single = CompiledModel.create(singlePath, options(config), env)
            val double = CompiledModel.create(doublePath, options(config), env)
            val sIn = single.createInputBuffers(); val sOut = single.createOutputBuffers()
            val dIn = double.createInputBuffers(); val dOut = double.createOutputBuffers()
            val rnd = java.util.Random(0)
            for (b in sIn + dIn) b.writeFloat(FloatArray(b.readFloat().size) { rnd.nextGaussian().toFloat() })
            single.run(sIn, sOut); double.run(dIn, dOut)   // warm-up (shader compile)
            repeat(2) { single.run(sIn, sOut); double.run(dIn, dOut) }
            for (rep in 1..reps) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                val stepMs = DoubleArray(steps)
                for (s in 0 until steps) {
                    val a = SystemClock.elapsedRealtimeNanos()
                    // wait for every block's output (a real pipeline feeds each block's output to the next one); without this the S23 GPU
                    // runtime deadlocked after about 25 s of back-to-back queued runs. The readback (about 11 MB) adds a few ms per block.
                    repeat(5) { double.run(dIn, dOut); dOut[0].readFloat() }    // 5 double-stream blocks
                    repeat(20) { single.run(sIn, sOut); sOut[0].readFloat() }   // 20 single-stream blocks
                    stepMs[s] = (SystemClock.elapsedRealtimeNanos() - a) / 1e6
                }
                val total = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                log("$tag rep=$rep total_ms=%.0f per_step_ms=%s".format(total, stepMs.joinToString("/") { "%.0f".format(it) }))
            }
            log("$tag RESULT done")
            single.close(); double.close()
        } catch (t: Throwable) {
            log("$tag FAILED ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    fun runFile(path: String, config: String, log: (String) -> Unit, holdSec: Int = 0) {
        val tag = "file=${path.substringAfterLast('/')} config=$config"
        try {
            fun memAvailMb(): Long = java.io.File("/proc/meminfo").readLines().first { it.startsWith("MemAvailable") }
                .filter { it.isDigit() }.toLong() / 1024
            System.gc()
            val availBefore = memAvailMb()   // system-wide, so GPU/ion buffers that are not in this process's PSS show up too
            val t0 = SystemClock.elapsedRealtimeNanos()
            val env = Environment.create()
            val model = CompiledModel.create(path, options(config), env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            val rnd = java.util.Random(0)
            for (b in inputs) b.writeFloat(FloatArray(b.readFloat().size) { rnd.nextGaussian().toFloat() })
            model.run(inputs, outputs)
            val first = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            log("$tag init+first_run_ms=%.0f inputs=%d".format(first, inputs.size))
            repeat(2) { model.run(inputs, outputs) }
            val ms = DoubleArray(5)
            for (i in ms.indices) {
                val a = SystemClock.elapsedRealtimeNanos()
                model.run(inputs, outputs)
                ms[i] = (SystemClock.elapsedRealtimeNanos() - a) / 1e6
            }
            val out = outputs[0].readFloat()
            val bad = out.count { it.isNaN() || it.isInfinite() }
            val mi = android.os.Debug.MemoryInfo()
            android.os.Debug.getMemoryInfo(mi)   // PSS in KB; graphics = GPU buffers mapped into this process
            log("$tag RESULT run_median_ms=%.1f run_min_ms=%.1f nan_or_inf=%d out_len=%d sys_mem_used_mb=%d pss_mb=%d graphics_mb=%s native_mb=%s".format(
                ms.sorted()[2], ms.min(), bad, out.size, availBefore - memAvailMb(), mi.totalPss / 1024,
                (mi.getMemoryStat("summary.graphics") ?: "?").toString().let { if (it == "?") it else (it.toInt() / 1024).toString() },
                (mi.getMemoryStat("summary.native-heap") ?: "?").toString().let { if (it == "?") it else (it.toInt() / 1024).toString() }))
            if (holdSec > 0) {   // keep the model loaded and busy so `adb shell dumpsys meminfo <pkg>` can be sampled
                log("$tag HOLDING ${holdSec}s")
                val end = SystemClock.elapsedRealtime() + holdSec * 1000L
                while (SystemClock.elapsedRealtime() < end) model.run(inputs, outputs)
                log("$tag HOLD_DONE")
            }
            model.close()
        } catch (t: Throwable) {
            log("$tag FAILED ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    private class Result(
        val initMs: Double, val runMedian: Double, val runP95: Double, val runMean: Double,
        val e2eMedian: Double, val output: FloatArray,
    )

    private fun options(config: String): CompiledModel.Options = when (config) {
        "cpu" -> CompiledModel.Options(Accelerator.CPU)
        "gpu" -> CompiledModel.Options(Accelerator.GPU)
        "gpu16" -> CompiledModel.Options(Accelerator.GPU).apply {
            gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16)
        }
        "gpu32" -> CompiledModel.Options(Accelerator.GPU).apply {
            gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
        }
        else -> throw IllegalArgumentException("unknown config $config")
    }

    private fun runOnce(ctx: Context, spec: Spec, config: String, input: FloatArray): FloatArray {
        val env = Environment.create()
        val model = CompiledModel.create(ctx.assets, spec.file, options(config), env)
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        val out = outputs[0].readFloat()
        model.close()
        return out
    }

    private fun bench(ctx: Context, spec: Spec, config: String, input: FloatArray, log: (String) -> Unit, tag: String): Result {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val env = Environment.create()
        val model = CompiledModel.create(ctx.assets, spec.file, options(config), env)
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()
        inputs[0].writeFloat(input)
        model.run(inputs, outputs) // first run includes shader compile / kernel setup
        val initMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
        log("$tag init+first_run_ms=%.0f".format(initMs))

        repeat(WARMUP) { model.run(inputs, outputs) }

        val runMs = DoubleArray(ITERS)
        val e2eMs = DoubleArray(ITERS)
        var last = FloatArray(0)
        for (i in 0 until ITERS) {
            val a = SystemClock.elapsedRealtimeNanos()
            inputs[0].writeFloat(input)
            val b = SystemClock.elapsedRealtimeNanos()
            model.run(inputs, outputs)
            val c = SystemClock.elapsedRealtimeNanos()
            last = outputs[0].readFloat()
            val d = SystemClock.elapsedRealtimeNanos()
            runMs[i] = (c - b) / 1e6
            e2eMs[i] = (d - a) / 1e6
        }
        // Sustained throughput: back-to-back write+run+read for SUSTAIN_SEC, reported per 5 s window
        // so thermal throttling shows up as a falling FPS.
        val windows = StringBuilder()
        val start = SystemClock.elapsedRealtime()
        var winStart = start
        var winCount = 0
        while (SystemClock.elapsedRealtime() - start < SUSTAIN_SEC * 1000L) {
            inputs[0].writeFloat(input)
            model.run(inputs, outputs)
            outputs[0].readFloat()
            winCount++
            val now = SystemClock.elapsedRealtime()
            if (now - winStart >= 5000) {
                windows.append("%.1f ".format(winCount * 1000.0 / (now - winStart)))
                winStart = now
                winCount = 0
            }
        }
        log("$tag sustained_fps_per_5s=${windows.toString().trim()}")
        model.close()
        val sorted = runMs.sorted()
        return Result(
            initMs, sorted[ITERS / 2], sorted[(ITERS * 0.95).toInt()], runMs.average(),
            e2eMs.sorted()[ITERS / 2], last,
        )
    }

    private fun loadInput(ctx: Context, spec: Spec): FloatArray {
        val raw = ctx.assets.open("bench_input.jpg").use { BitmapFactory.decodeStream(it) }
        val bmp = Bitmap.createScaledBitmap(raw, spec.size, spec.size, true)
        val px = IntArray(spec.size * spec.size)
        bmp.getPixels(px, 0, spec.size, 0, 0, spec.size, spec.size)
        val out = FloatArray(px.size * spec.channels)
        for (i in px.indices) {
            val p = px[i]
            out[i * spec.channels + 0] = ((p shr 16) and 0xFF) / 255f
            out[i * spec.channels + 1] = ((p shr 8) and 0xFF) / 255f
            out[i * spec.channels + 2] = (p and 0xFF) / 255f
            // 4th channel (hair model) stays 0 = "no previous mask"
        }
        return out
    }
}
