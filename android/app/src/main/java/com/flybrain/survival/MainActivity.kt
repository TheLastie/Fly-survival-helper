package com.flybrain.survival

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : Activity() {
    private var fileCallback: ValueCallback<Array<Uri>>? = null
    private val FILE_REQ = 1001
    private val PORT = 8321
    private var status: TextView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val lay = LinearLayout(this)
        lay.orientation = LinearLayout.VERTICAL
        lay.gravity = Gravity.CENTER
        lay.setBackgroundColor(Color.rgb(20, 23, 28))
        status = TextView(this)
        status!!.text = "FlyBrain Survival\n\nзагрузка базы (10–20 с при первом запуске)…"
        status!!.setTextColor(Color.rgb(223, 229, 236))
        status!!.textSize = 16f
        status!!.gravity = Gravity.CENTER
        lay.addView(status)
        setContentView(lay)

        // Python-рантайм в фоновом потоке: UI не блокируется, нет ANR
        Thread {
            try {
                if (!Python.isStarted()) Python.start(AndroidPlatform(this))
                status?.post { status?.text = "индексация и старт сервера…" }
                Python.getInstance().getModule("android_host")
                    .callAttr("start", filesDir.absolutePath)
                status?.post { showWeb() }
            } catch (t: Throwable) {
                status?.post { showFatal("Chaquopy/Python: " + t.message, t) }
            }
        }.start()
    }

    // Экран ошибки с кнопкой «отправить лог»: трейсбэк улетает в мессенджер
    // одним нажатием — компьютер и adb не нужны.
    private fun showFatal(where: String, t: Throwable) {
        val log = java.io.File(filesDir, "crash.log")
        val details = if (log.exists()) log.readText() else t.stackTraceToString()
        val full = "$where\n\n$details"
        val lay = LinearLayout(this)
        lay.orientation = LinearLayout.VERTICAL
        lay.gravity = Gravity.CENTER
        lay.setBackgroundColor(Color.rgb(20, 23, 28))
        val tv = TextView(this)
        tv.text = "FlyBrain: ошибка запуска\n\n$where\n\n(полный лог — по кнопке ниже)"
        tv.setTextColor(Color.rgb(223, 229, 236))
        tv.textSize = 15f
        tv.gravity = Gravity.CENTER
        lay.addView(tv)
        val btn = Button(this)
        btn.text = "Отправить лог"
        btn.setOnClickListener {
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_SUBJECT, "FlySurvival crash log")
                putExtra(Intent.EXTRA_TEXT, full.take(8000))
            }
            startActivity(Intent.createChooser(send, "Отправить лог"))
        }
        lay.addView(btn)
        setContentView(lay)
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun showWeb() {
        val web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.settings.allowFileAccess = false
        web.webViewClient = object : WebViewClient() {
            private var attempts = 0
            override fun onReceivedError(v: WebView, code: Int, desc: String, url: String) {
                if (attempts < 30 && url.contains("127.0.0.1")) {
                    attempts++
                    v.postDelayed({ v.reload() }, 1000)
                }
            }
        }
        web.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                v: WebView, cb: ValueCallback<Array<Uri>>, p: FileChooserParams
            ): Boolean {
                fileCallback?.onReceiveValue(null)
                fileCallback = cb
                startActivityForResult(p.createIntent(), FILE_REQ)
                return true
            }
        }
        setContentView(web)
        web.loadUrl("http://127.0.0.1:$PORT")
    }

    override fun onActivityResult(req: Int, res: Int, data: Intent?) {
        super.onActivityResult(req, res, data)
        if (req == FILE_REQ) {
            fileCallback?.onReceiveValue(
                WebChromeClient.FileChooserParams.parseResult(res, data))
            fileCallback = null
        }
    }

    override fun onDestroy() {
        Thread {
            try { Python.getInstance().getModule("android_host").callAttr("stop") } catch (_: Exception) {}
        }.start()
        super.onDestroy()
    }
}
