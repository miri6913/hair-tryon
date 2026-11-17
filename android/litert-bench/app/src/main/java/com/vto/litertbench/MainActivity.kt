package com.vto.litertbench

import android.app.Activity
import android.graphics.Typeface
import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView

/**
 * Headless benchmark launcher. One launch = one (model, config) pair, so a native crash in one
 * accelerator does not hide the others.
 *
 *   adb shell am start -n com.vto.litertbench/.MainActivity --es model hair --es config gpu16
 *
 * model:  hair | multi
 * config: cpu | gpu | gpu16 | gpu32
 * Results are logged under tag VTO_BENCH and shown on screen.
 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val view = TextView(this).apply {
            textSize = 13f
            typeface = Typeface.MONOSPACE
            setPadding(24, 24, 24, 24)
        }
        setContentView(ScrollView(this).apply { addView(view) })

        val model = intent.getStringExtra("model") ?: "hair"
        val config = intent.getStringExtra("config") ?: "cpu"
        val file = intent.getStringExtra("file")   // spike: absolute path of any .tflite, e.g. /sdcard/Android/data/<pkg>/files/x.tflite
        val sim = intent.getStringExtra("sim")      // spike 3: "<single path>,<double path>" -> simulated denoising loop
        val hold = intent.getIntExtra("hold", 0)    // spike 3: seconds to keep a --es file model busy (memory sampling)
        Thread {
            if (sim != null) {
                val (sp, dp) = sim.split(",")
                Bench.runSim(sp, dp, intent.getIntExtra("steps", 2), intent.getIntExtra("reps", 3), config) { line ->
                    Log.i(TAG, line)
                    runOnUiThread { view.append(line + "\n") }
                }
                return@Thread
            }
            if (file != null) {
                Bench.runFile(file, config, { line ->
                    Log.i(TAG, line)
                    runOnUiThread { view.append(line + "\n") }
                }, hold)
                return@Thread
            }
            Bench.run(this, model, config) { line ->
                Log.i(TAG, line)
                runOnUiThread { view.append(line + "\n") }
            }
        }.start()
    }

    companion object {
        const val TAG = "VTO_BENCH"
    }
}
