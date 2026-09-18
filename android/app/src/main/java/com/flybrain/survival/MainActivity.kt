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
        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        Thread {
            Python.getInstance().getModule("android_host").callAttr("start", filesDir.absolutePath)
        }.start()

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
