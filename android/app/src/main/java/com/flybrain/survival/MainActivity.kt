package com.flybrain.survival

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.app.Activity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : Activity() {
    private var fileCallback: ValueCallback<Array<Uri>>? = null
    private val FILE_REQ = 1001
    private val PORT = 8321

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Python-рантайм + наш сервер в фоновом потоке
        try {
            if (!Python.isStarted()) Python.start(AndroidPlatform(this))
            Thread {
                try {
                    Python.getInstance().getModule("android_host")
                        .callAttr("start", filesDir.absolutePath)
                } catch (t: Throwable) {
                    runOnUiThread { showError("Python: " + t.message) }
                }
            }.start()
        } catch (t: Throwable) {
            showError("Chaquopy: " + t.message)
        }

        val web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.settings.allowFileAccess = false
        web.webViewClient = WebViewClient()
        // выбор фото/камеры из <input type=file> Web-UI
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
        // Серверу нужно 3–10 с (распаковка состояния из assets + импорт
        // модулей). WebView перезагружает страницу до победного, максимум
        // ~30 попыток с интервалом 1 с; если сервер так и не поднялся —
        // аварийный экран android_host покажет traceback.
        web.webViewClient = object : WebViewClient() {
            private var attempts = 0
            override fun onReceivedError(
                v: WebView, code: Int, desc: String, url: String
            ) {
                if (attempts < 30 && url.contains("127.0.0.1")) {
                    attempts++
                    v.postDelayed({ v.reload() }, 1000)
                }
            }
        }
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

    private fun showError(msg: String) {
        android.app.AlertDialog.Builder(this)
            .setTitle("FlyBrain: ошибка запуска")
            .setMessage(msg + "\n\nДетали будут на экране приложения.")
            .setPositiveButton("OK", null)
            .show()
    }

    override fun onDestroy() {
        Thread {
            try { Python.getInstance().getModule("android_host").callAttr("stop") } catch (_: Exception) {}
        }.start()
        super.onDestroy()
    }
}
