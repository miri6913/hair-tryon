package com.vto.hairtryon.gl

import android.content.Context
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import androidx.camera.core.SurfaceRequest
import androidx.core.content.ContextCompat
import com.vto.hairtryon.palette.Palette
import com.vto.hairtryon.palette.Shade
import com.vto.hairtryon.seg.LiteRtSegmenter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Per-frame pipeline (all on the GL thread; segmentation runs on its own thread):
 *   camera OES -> upright RGB FBO
 *     -> 512x512 readback -> LiteRT GPU segmentation -> raw mask texture -> temporal EMA -> smoothed mask
 *     -> recolor: prep (L*m, m) -> separable blur -> recolor shader -> result FBO
 *   -> present (center-crop, before/after split) to the screen.
 */
class CameraRenderer(private val ctx: Context, private val glView: GLSurfaceView) : GLSurfaceView.Renderer {
    // ---- state written from the UI thread ----
    @Volatile var showMask = false
    @Volatile var flipH = false
    @Volatile var refine = true                  // guided-filter edge refinement of the mask
    @Volatile var gfRadius = 2                   // box radius in quarter-res texels (~8 px at full res, as in Phase 0)
    @Volatile var gfEps = 0.002f
    // chosen by scripts/smoothing_sweep.py: 4.5x steadier edge than raw with ~no tracking loss on a sliding photo
    @Volatile var emaAlpha = 0.12f
    @Volatile var emaMotionGain = 3.0f
    var usePrior: Boolean
        get() = segmenter.usePrior
        set(v) { segmenter.usePrior = v }
    @Volatile var intensity = 0f                 // 0 = original image
    @Volatile var shade: Shade = Palette.shades[3]
    @Volatile var split = 0f                     // 0 = all recolored .. 1 = all original

    // ---- read by the UI ----
    @Volatile var rotationDegrees = 0
    @Volatile var mirror = false
    @Volatile var fps = 0f
    val segMs: Float get() = segmenter.lastMs
    val segFps: Float get() = segmenter.fps

    private val segmenter = LiteRtSegmenter(ctx)

    private var oesTex = 0
    private var surfaceTexture: SurfaceTexture? = null
    private val frameAvailable = AtomicBoolean(false)
    private var pendingRequest: SurfaceRequest? = null
    private var camW = 0
    private var camH = 0
    private var haveFrame = false
    private val captureCallback = AtomicReference<((Bitmap) -> Unit)?>(null)
    private var stillBitmap: Bitmap? = null
    private var stillTex = 0
    private val stMatrix = FloatArray(16)
    private var stLogged = false

    private lateinit var oes: Program
    private lateinit var blit: Program
    private lateinit var ema: Program
    private lateinit var prep: Program
    private lateinit var blur: Program
    private lateinit var recolor: Program
    private lateinit var present: Program
    private lateinit var gfPrep: Program
    private lateinit var box: Program
    private lateinit var gfCoef: Program

    private var upright: Fbo? = null
    private var result: Fbo? = null
    private var halfA: Fbo? = null
    private var halfB: Fbo? = null
    private var gfA: Fbo? = null      // 32-bit float, half res: moments
    private var gfB: Fbo? = null
    private var coef: Fbo? = null     // 16-bit float, half res, linear: final (mean a, mean b)
    private var viewW = 1
    private var viewH = 1

    // segmentation plumbing
    private var segFbo: Fbo? = null
    private val readBuf: ByteBuffer =
        ByteBuffer.allocateDirect(LiteRtSegmenter.SIZE * LiteRtSegmenter.SIZE * 4).order(ByteOrder.nativeOrder())
    private var maskTex = 0

    // latency instrumentation: how old (in camera time) is the mask relative to the frame being displayed
    private var curFrameTs = 0L
    private var uploadedMaskTs = 0L
    private var lagSumMs = 0.0
    private var lagCount = 0
    private var readbackSumMs = 0.0
    private var readbackCount = 0
    private var frameSumMs = 0.0
    private var frameCount = 0

    // temporal smoothing (ping-pong): smooth[smoothIdx] holds the latest smoothed probability
    private var smooth: Array<Fbo>? = null
    private var smoothIdx = 0

    // the person's hair lightness quantiles / mean chroma, eased toward the segmenter's latest measurement
    private var srcQ: FloatArray? = null
    private var meanA = 0f
    private var meanB = 0f

    private var fpsWindowStart = 0L
    private var fpsFrames = 0

    /** Called by CameraX on the main thread; hops to the GL thread to attach our SurfaceTexture. */
    fun provideSurface(request: SurfaceRequest) {
        glView.queueEvent {
            pendingRequest?.willNotProvideSurface()
            pendingRequest = request
            trySetupSurface()
        }
    }

    private fun trySetupSurface() {
        val st = surfaceTexture ?: return
        val request = pendingRequest ?: return
        pendingRequest = null
        camW = request.resolution.width
        camH = request.resolution.height
        st.setDefaultBufferSize(camW, camH)
        val surface = Surface(st)
        val executor = ContextCompat.getMainExecutor(ctx)
        request.provideSurface(surface, executor) { surface.release() }
        request.setTransformationInfoListener(executor) { info ->
            rotationDegrees = info.rotationDegrees
            mirror = info.isMirroring
            Log.i(TAG, "camera ${camW}x$camH rotation=${info.rotationDegrees} mirroring=${info.isMirroring}")
        }
        haveFrame = false
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val t = IntArray(1)
        GLES30.glGenTextures(1, t, 0)
        oesTex = t[0]
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        surfaceTexture = SurfaceTexture(oesTex).also { st ->
            st.setOnFrameAvailableListener {
                frameAvailable.set(true)
                glView.requestRender()
            }
        }
        oes = Program(GlUtil.program(ctx, "quad.vert", "oes_upright.frag"))
        blit = Program(GlUtil.program(ctx, "quad.vert", "blit.frag"))
        ema = Program(GlUtil.program(ctx, "quad.vert", "mask_ema.frag"))
        prep = Program(GlUtil.program(ctx, "quad.vert", "recolor_prep.frag"))
        blur = Program(GlUtil.program(ctx, "quad.vert", "blur.frag"))
        recolor = Program(GlUtil.program(ctx, "quad.vert", "recolor.frag"))
        present = Program(GlUtil.program(ctx, "quad.vert", "present.frag"))
        gfPrep = Program(GlUtil.program(ctx, "quad.vert", "gf_prep.frag"))
        box = Program(GlUtil.program(ctx, "quad.vert", "box.frag"))
        gfCoef = Program(GlUtil.program(ctx, "quad.vert", "gf_coef.frag"))
        upright = null
        result = null
        halfA = null
        halfB = null
        gfA = null
        gfB = null
        coef = null
        haveFrame = false
        stillTex = 0
        uploadStill() // the GL context is recreated on resume: re-upload the still image, if any

        // segmentation resources
        val n = LiteRtSegmenter.SIZE
        segFbo = Fbo(n, n)
        val m = IntArray(1)
        GLES30.glGenTextures(1, m, 0)
        maskTex = m[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTex)
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_R16F, n, n)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        val empty = ByteBuffer.allocateDirect(n * n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (i in 0 until n * n) empty.put(i, -10f) // "no hair" until the first mask arrives
        GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, n, n, GLES30.GL_RED, GLES30.GL_FLOAT, empty)

        smooth = arrayOf(Fbo(n, n, GLES30.GL_RGBA16F), Fbo(n, n, GLES30.GL_RGBA16F)).also { pair ->
            for (f in pair) {
                f.bind()
                GLES30.glClearColor(0f, 0f, 0f, 1f)
                GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
            }
        }
        smoothIdx = 0

        Log.i(TAG, "GL ready: ${GLES30.glGetString(GLES30.GL_RENDERER)} / ${GLES30.glGetString(GLES30.GL_VERSION)}")
        trySetupSurface()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewW = width
        viewH = height
        GLES30.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        val frameStartNs = SystemClock.elapsedRealtimeNanos()
        val st = surfaceTexture ?: return
        val still = stillBitmap != null
        if (still) curFrameTs = SystemClock.elapsedRealtimeNanos()
        if (!still && frameAvailable.getAndSet(false)) {
            st.updateTexImage()
            st.getTransformMatrix(stMatrix)
            curFrameTs = st.timestamp
            if (!stLogged) {
                stLogged = true
                Log.i(TAG, "ST matrix: " + stMatrix.joinToString(" ") { "%.2f".format(it) })
            }
            haveFrame = true
        }
        if (!haveFrame) {
            GLES30.glClearColor(0f, 0f, 0f, 1f)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
            return
        }

        val frame = ensureTargets(rotationDegrees)

        // Pass 1: camera buffer -> upright RGB texture.
        // The SurfaceTexture matrix from the camera service already carries the sensor-orientation rotation
        // (observed: a 90 degree rotation for a 270 degree sensor in portrait), so we do NOT rotate again by
        // CameraX's rotationDegrees. Revisit for non-natural display rotations (tablet).
        frame.bind()
        if (still) {
            // a Bitmap uploads top-row-first, the upright FBO convention is bottom-row-first
            drawBlit(stillTex, 1f, 1f, flipY = true, overlay = false)
        } else {
            oes.use()
            oes.tex("uCam", 0, oesTex, GLES11Ext.GL_TEXTURE_EXTERNAL_OES)
            val uvMat = (if (flipH) Affine.mirrorU else Affine()).then(Affine.fromSurfaceTexture(stMatrix))
            oes.mat3("uUvMat", uvMat.toMat3())
            GlUtil.drawQuad()
        }

        uploadMaskIfReady()
        if (uploadedMaskTs != 0L && curFrameTs != 0L) {
            lagSumMs += (curFrameTs - uploadedMaskTs) / 1e6
            lagCount++
        }
        driveSegmentation(frame)
        updateHairStats()

        val recolored = intensity > 0.002f
        if (recolored) renderRecolor(frame)

        val viewAspect = viewW.toFloat() / viewH
        val imgAspect = frame.width.toFloat() / frame.height
        val (sx, sy) = if (viewAspect > imgAspect) 1f to imgAspect / viewAspect else viewAspect / imgAspect to 1f

        // still mode saves the whole photo (no screen crop) so a golden comparison needs no crop bookkeeping
        captureCallback.getAndSet(null)?.let { cb ->
            if (still) captureFrame(frame, recolored, 1f, 1f, cb) else captureFrame(frame, recolored, sx, sy, cb)
        }

        // Final pass to the screen
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, viewW, viewH)
        if (showMask) {
            drawBlit(frame.tex, sx, sy, flipY = false, overlay = true)
        } else {
            present.use()
            present.tex("uOrig", 0, frame.tex)
            present.tex("uResult", 1, if (recolored) result!!.tex else frame.tex)
            present.f2("uCropScale", sx, sy)
            present.f("uSplit", if (recolored) split else 1f)
            GlUtil.drawQuad()
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        }
        GlUtil.check("frame")

        if (MEASURE_GPU) {
            GLES30.glFinish() // debug only: serializes the pipeline to measure the true per-frame cost
            frameSumMs += (SystemClock.elapsedRealtimeNanos() - frameStartNs) / 1e6
            frameCount++
        }
        countFrame()
    }

    /**
     * Renders the clean result (no compare divider, mask overlay or UI) with the same center-crop as the screen
     * into an offscreen target and reads it back. [cb] gets a top-row-first ARGB bitmap on the GL thread.
     */
    private fun captureFrame(frame: Fbo, recolored: Boolean, sx: Float, sy: Float, cb: (Bitmap) -> Unit) {
        val w = (frame.width * sx).toInt().coerceAtLeast(1)
        val h = (frame.height * sy).toInt().coerceAtLeast(1)
        val target = Fbo(w, h)
        try {
            target.bind()
            present.use()
            present.tex("uOrig", 0, frame.tex)
            present.tex("uResult", 1, if (recolored) result!!.tex else frame.tex)
            present.f2("uCropScale", sx, sy)
            present.f("uSplit", if (recolored) 0f else 1f)
            GlUtil.drawQuad()
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            val buf = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())
            GLES30.glReadPixels(0, 0, w, h, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buf)
            GlUtil.check("capture")
            // glReadPixels is bottom-row first: flip while copying
            val flipped = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())
            val row = ByteArray(w * 4)
            for (y in 0 until h) {
                buf.position((h - 1 - y) * w * 4)
                buf.get(row)
                flipped.put(row)
            }
            flipped.rewind()
            val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(flipped)
            cb(bmp)
        } finally {
            target.release()
        }
    }

    /**
     * The temporally smoothed hair probability (before guided-filter refinement and mask contrast) as
     * LiteRtSegmenter.SIZE^2 floats, top row first. GL thread only: call it from a [requestCapture] callback.
     */
    fun readSmoothedMask(): FloatArray {
        val n = LiteRtSegmenter.SIZE
        val src = smooth!![smoothIdx]
        src.bind()
        val buf = ByteBuffer.allocateDirect(n * n * 4 * 4).order(ByteOrder.nativeOrder())
        GLES30.glReadPixels(0, 0, n, n, GLES30.GL_RGBA, GLES30.GL_FLOAT, buf)
        GlUtil.check("readSmoothedMask")
        val fb = buf.asFloatBuffer()
        return FloatArray(n * n) { fb.get(it * 4) }
    }

    /** Asks for a clean capture of the next rendered frame; [cb] runs on the GL thread. */
    fun requestCapture(cb: (Bitmap) -> Unit) {
        captureCallback.set(cb)
        glView.requestRender()
    }

    /** Still-image debug mode: use [bitmap] as the frame source instead of the camera (golden tests). */
    fun setStill(bitmap: Bitmap) {
        glView.queueEvent {
            stillBitmap = bitmap
            uploadStill()
        }
    }

    /** Back to the live camera after [setStill] (the camera stays bound and keeps delivering frames). */
    fun clearStill() {
        glView.queueEvent { stillBitmap = null }
    }

    private fun uploadStill() {
        val bmp = stillBitmap ?: return
        if (stillTex == 0) {
            val t = IntArray(1)
            GLES30.glGenTextures(1, t, 0)
            stillTex = t[0]
        }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, stillTex)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bmp, 0)
        camW = bmp.width
        camH = bmp.height
        rotationDegrees = 0
        haveFrame = true
        glView.requestRender()
    }

    private fun renderRecolor(frame: Fbo) {
        val a = halfA ?: return
        val b = halfB ?: return
        val res = result ?: return
        val mask = smooth!![smoothIdx].tex
        val coefTex = coef?.tex ?: return
        val doRefine = refine
        if (doRefine) renderGuidedFilter(frame, mask)

        // (L*m, m) at half resolution
        a.bind()
        prep.use()
        prep.tex("uRgb", 0, frame.tex)
        prep.tex("uMask", 1, mask)
        prep.tex("uCoef", 2, coefTex)
        prep.f("uRefine", if (doRefine) 1f else 0f)
        GlUtil.drawQuad()

        // separable blur: horizontal a -> b, vertical b -> a (sigma 2 px at half res == 4 px at full, as in Phase 0)
        b.bind()
        blur.use()
        blur.tex("uSrc", 0, a.tex)
        blur.f2("uStep", 1f / a.width, 0f)
        blur.f("uSigma", 2f)
        GlUtil.drawQuad()
        a.bind()
        blur.tex("uSrc", 0, b.tex)
        blur.f2("uStep", 0f, 1f / a.height)
        GlUtil.drawQuad()

        // recolor at full working resolution
        res.bind()
        recolor.use()
        recolor.tex("uRgb", 0, frame.tex)
        recolor.tex("uMask", 1, mask)
        recolor.tex("uBlur", 2, a.tex)
        recolor.tex("uCoef", 3, coefTex)
        recolor.f("uRefine", if (doRefine) 1f else 0f)
        val sh = shade
        // until the first measurement, an identity map (person's quantiles == shade's) keeps the frame natural
        val q = Palette.capSlope(srcQ ?: sh.l, sh.l)
        recolor.fv("uSrcQ", q)
        recolor.fv("uDstQ", sh.l)
        recolor.f2("uAbShadow", sh.abShadow)
        recolor.f2("uAbMid", sh.abMid)
        recolor.f2("uAbHigh", sh.abHigh)
        recolor.f2("uGain", sh.gainA, sh.gainB)
        recolor.f("uKeep", sh.keepChroma)
        recolor.f2("uMeanAB", meanA, meanB)
        recolor.f("uIntensity", intensity)
        recolor.f("uDetailGain", 1.0f)
        recolor.f("uEps", 12.0f)
        GlUtil.drawQuad()
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
    }

    /**
     * Fast guided filter (He et al.) of the smoothed mask with the camera luma as guide, at half resolution:
     * moments -> box blur -> (a, b) -> box blur. The recolor shaders evaluate q = a*I + b at full resolution,
     * which snaps the mask edge to the real hair edge in the frame.
     */
    private fun renderGuidedFilter(frame: Fbo, mask: Int) {
        val ga = gfA ?: return
        val gb = gfB ?: return
        val out = coef ?: return
        val r = gfRadius

        ga.bind()
        gfPrep.use()
        gfPrep.tex("uRgb", 0, frame.tex)
        gfPrep.tex("uMask", 1, mask)
        gfPrep.f2("uTexel", 1f / frame.width, 1f / frame.height)
        GlUtil.drawQuad()

        boxPass(ga.tex, gb, 1f / ga.width, 0f, r)       // moments: horizontal  ga -> gb
        boxPass(gb.tex, ga, 0f, 1f / ga.height, r)      //          vertical    gb -> ga

        gb.bind()
        gfCoef.use()
        gfCoef.tex("uMoments", 0, ga.tex)
        gfCoef.f("uEps", gfEps)
        GlUtil.drawQuad()

        boxPass(gb.tex, ga, 1f / gb.width, 0f, r)       // smooth (a, b): horizontal gb -> ga
        boxPass(ga.tex, out, 0f, 1f / ga.height, r)     //                vertical   ga -> coef
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
    }

    private fun boxPass(src: Int, dst: Fbo, stepX: Float, stepY: Float, radius: Int) {
        dst.bind()
        box.use()
        box.tex("uSrc", 0, src)
        box.f2("uStep", stepX, stepY)
        box.i("uRadius", radius)
        GlUtil.drawQuad()
    }

    private fun drawBlit(tex: Int, sx: Float, sy: Float, flipY: Boolean, overlay: Boolean) {
        blit.use()
        blit.tex("uTex", 0, tex)
        blit.tex("uMask", 1, smooth?.get(smoothIdx)?.tex ?: maskTex)
        blit.f2("uCropScale", sx, sy)
        blit.f("uFlipY", if (flipY) 1f else 0f)
        blit.f("uOverlay", if (overlay) 1f else 0f)
        GlUtil.drawQuad()
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
    }

    /** Uploads a freshly finished raw mask and folds it into the temporally smoothed mask. */
    private fun uploadMaskIfReady() {
        val mask = segmenter.takeMask() ?: return
        val n = LiteRtSegmenter.SIZE
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTex)
        GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, n, n, GLES30.GL_RED, GLES30.GL_FLOAT, mask)
        uploadedMaskTs = segmenter.maskTimestampNs

        val pair = smooth ?: return
        val src = pair[smoothIdx]
        val dst = pair[1 - smoothIdx]
        dst.bind()
        ema.use()
        ema.tex("uRaw", 0, maskTex)
        ema.tex("uPrev", 1, src.tex)
        ema.f("uAlpha", emaAlpha)
        ema.f("uMotionGain", emaMotionGain)
        GlUtil.drawQuad()
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        smoothIdx = 1 - smoothIdx
    }

    /** Eases the person's hair statistics toward the segmenter's latest measurement (they arrive ~7 times/s). */
    private fun updateHairStats() {
        val m = segmenter.hairQuantiles ?: return
        val cur = srcQ
        if (cur == null) {
            srcQ = m.copyOf()
            meanA = segmenter.hairA
            meanB = segmenter.hairB
        } else {
            for (i in 0 until 5) cur[i] += (m[i] - cur[i]) * 0.15f
            meanA += (segmenter.hairA - meanA) * 0.15f
            meanB += (segmenter.hairB - meanB) * 0.15f
        }
    }

    /**
     * Latest-frame-wins feed of the segmenter. When the worker is idle, downscale the current frame to 512x512 and
     * read it back synchronously: it stalls the GL thread for a moment, but the mask is then computed from the
     * frame being shown, one frame fresher than the earlier async-PBO design.
     */
    private fun driveSegmentation(upright: Fbo) {
        if (!segmenter.isIdle) return
        val n = LiteRtSegmenter.SIZE
        val seg = segFbo ?: return
        val t0 = SystemClock.elapsedRealtimeNanos()
        seg.bind()
        drawBlit(upright.tex, 1f, 1f, flipY = true, overlay = false) // readback is bottom-row first, model wants top-row first
        readBuf.rewind()
        GLES30.glReadPixels(0, 0, n, n, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, readBuf)
        readbackSumMs += (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
        readbackCount++
        segmenter.submit(readBuf, curFrameTs)
    }

    /** (Re)creates the upright / result / half-resolution targets when the camera size or rotation changes. */
    private fun ensureTargets(rot: Int): Fbo {
        val (w, h) = if (rot % 180 == 0) camW to camH else camH to camW
        val scale = minOf(1f, MAX_LONG_SIDE.toFloat() / maxOf(w, h))
        val tw = (w * scale).toInt()
        val th = (h * scale).toInt()
        val cur = upright
        if (cur != null && cur.width == tw && cur.height == th) return cur
        cur?.release()
        result?.release()
        halfA?.release()
        halfB?.release()
        gfA?.release()
        gfB?.release()
        coef?.release()
        result = Fbo(tw, th)
        halfA = Fbo(tw / 2, th / 2, GLES30.GL_RGBA16F)
        halfB = Fbo(tw / 2, th / 2, GLES30.GL_RGBA16F)
        gfA = Fbo(tw / 4, th / 4, GLES30.GL_RGBA32F, linear = false)
        gfB = Fbo(tw / 4, th / 4, GLES30.GL_RGBA32F, linear = false)
        coef = Fbo(tw / 4, th / 4, GLES30.GL_RGBA16F)
        return Fbo(tw, th).also {
            upright = it
            Log.i(TAG, "targets ${tw}x$th (camera ${camW}x$camH, rotation $rot)")
        }
    }

    private fun countFrame() {
        val now = SystemClock.elapsedRealtime()
        if (fpsWindowStart == 0L) fpsWindowStart = now
        fpsFrames++
        if (now - fpsWindowStart >= 1000) {
            fps = fpsFrames * 1000f / (now - fpsWindowStart)
            Log.i(
                "VTO_FPS",
                "render_fps=%.1f frame_ms=%.1f mask_lag_ms=%.0f readback_ms=%.1f refine=%b prior=%b edge_flicker=%.4f hairL_p50=%.1f hairPx=%d".format(
                    fps, if (frameCount > 0) frameSumMs / frameCount else 0.0,
                    if (lagCount > 0) lagSumMs / lagCount else 0.0,
                    if (readbackCount > 0) readbackSumMs / readbackCount else 0.0, refine, segmenter.usePrior,
                    segmenter.edgeFlicker, srcQ?.get(2) ?: Float.NaN, segmenter.hairPixels,
                ),
            )
            lagSumMs = 0.0; lagCount = 0; readbackSumMs = 0.0; readbackCount = 0; frameSumMs = 0.0; frameCount = 0
            fpsWindowStart = now
            fpsFrames = 0
        }
    }

    fun release() = segmenter.close()

    companion object {
        private const val TAG = "VTO_RENDER"
        private const val MAX_LONG_SIDE = 1280

        /** Debug: glFinish every frame and log the total frame time (distorts pipelining, so keep off in release). */
        private const val MEASURE_GPU = false
    }
}
