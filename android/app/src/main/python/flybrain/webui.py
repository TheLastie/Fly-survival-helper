"""Web UI (P8): локальный интерфейс поверх FlySystem. Только stdlib.

Запуск:  python webui.py [--state book_state] [--port 8321] [--host 127.0.0.1]

Один HTML-файл внутри модуля, vanilla JS. Сервер однопользовательский
(как CLI): состояние системы разделяется между запросами, last_result
используется фидбеком. Наружу ничего не уходит — слушает localhost.
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Мозг мухи — поиск</title>
<style>
  :root { --bg:#14171c; --panel:#1c2129; --fg:#dfe5ec; --dim:#8b95a3;
          --acc:#5ec1ff; --ok:#7ddb8a; --warn:#e8c05a; --bad:#e87d7d; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 system-ui,sans-serif; display:flex; min-height:100vh; }
  #side { width:250px; padding:16px; background:var(--panel); flex-shrink:0; }
  #side h2 { font-size:13px; color:var(--dim); text-transform:uppercase; }
  #main { flex:1; padding:24px 32px; max-width:900px; }
  h1 { font-size:20px; } h1 small { color:var(--dim); font-weight:normal; }
  #q { width:100%; padding:12px 14px; font-size:16px; border-radius:8px;
       border:1px solid #333a45; background:#0e1116; color:var(--fg); }
  #q:focus { outline:none; border-color:var(--acc); }
  .res { background:var(--panel); border-radius:10px; padding:14px 16px;
         margin:12px 0; border-left:3px solid var(--dim); }
  .res.confident { border-left-color:var(--ok); }
  .res.unsure { border-left-color:var(--warn); }
  .res.unknown { border-left-color:var(--bad); }
  .meta { color:var(--dim); font-size:12px; margin-top:8px; }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px;
           font-size:12px; background:#2a3340; margin-right:6px; }
  .answer { font-size:17px; margin-bottom:4px; }
  button.fb { background:#2a3340; color:var(--fg); border:none; border-radius:6px;
              padding:4px 10px; margin-right:6px; cursor:pointer; font-size:13px; }
  button.fb:hover { background:#37414f; }
  #extracted { background:#12303f; border:1px solid #1d5a75; border-radius:8px;
               padding:10px 14px; margin:12px 0; }
  input.cmd { width:100%; padding:6px 8px; margin:4px 0; background:#0e1116;
              border:1px solid #333a45; color:var(--fg); border-radius:6px; }
  .src { font-size:13px; color:var(--dim); overflow:hidden;
         text-overflow:ellipsis; white-space:nowrap; }
</style>
</head>
<body>
<div id="side">
  <h2>Источники</h2>
  <div id="sources">—</div>
  <h2>Команды</h2>
  <input class="cmd" id="idx" placeholder="индексировать: путь" title="Enter — индексировать">
  <input class="cmd" id="mem" placeholder="запомнить: факт" title="Enter — запомнить">
  <input class="cmd" id="sav" placeholder="сохранить: путь" title="Enter — сохранить">
  <div class="meta" id="stat"></div>
  <h2>Распознавание</h2>
  <input type="file" id="photo" accept="image/*" style="width:100%;font-size:12px">
  <div id="rec_out" style="font-size:13px;margin-top:6px"></div>
  <h2>Виды</h2>
  <div id="species">—</div>
</div>
<div id="main">
  <h1>Мозг мухи <small>локальная поисковая система</small></h1>
  <input id="q" placeholder="вопрос или поисковый запрос… (Enter)" autofocus>
  <div id="extracted" style="display:none"></div>
  <div id="results"></div>
</div>
<script>
const $ = id => document.getElementById(id);
async function api(path, opts) {
  const r = await fetch(path, opts && {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(opts)});
  return r.json();
}
async function search() {
  const q = $('q').value.trim();
  if (!q) return;
  $('results').innerHTML = '<div class="meta">ищу…</div>';
  $('extracted').style.display = 'none';
  const d = await api('/api/search?q=' + encodeURIComponent(q));
  if (d.extracted) {
    $('extracted').innerHTML = '<b>Ответ:</b> ' + esc(d.extracted.answer) +
      '<div class="meta">«' + esc(d.extracted.evidence.slice(0,160)) + '» · уверенность ' +
      Math.round(d.extracted.conf*100) + '%</div>';
    $('extracted').style.display = 'block';
  }
  $('results').innerHTML = d.candidates.map((c,i) =>
    '<div class="res ' + d.zone + '">' +
    '<div class="answer">' + esc(c.text.slice(0, 260)) + (c.text.length>260?'…':'') + '</div>' +
    '<div class="meta"><span class="badge">conf ' + Math.round(c.conf*100) + '%</span>' +
    '<span class="badge">' + d.zone + '</span>' + esc(c.citation) + '</div>' +
    '<div style="margin-top:6px">' +
    '<button class="fb" onclick="fb(1,'+c.idx+',this)">верно</button>' +
    '<button class="fb" onclick="fb(-1,'+c.idx+',this)">забудь</button>' +
    '</div></div>').join('') ||
    '<div class="res unknown"><div class="answer">не помню ничего похожего.</div>' +
    '<div class="meta">знакомость ' + d.familiarity.toFixed(2) + '</div></div>';
  $('stat').textContent = 'знакомость ' + d.familiarity.toFixed(2) + ' · ' + d.n + ' эпизодов';
  $('photo').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    // превью снимка — пользователь видит, ЧТО распознавалось
    $('rec_out').innerHTML = '<img src="' + r.result + '" style="max-width:100%;border-radius:8px"><div class="meta" style="margin-top:4px">распознаю…</div>';
    api('/api/recognize', {b64: r.result.split(',')[1], name: f.name})
    .then(d => {
      const head = '<img src="' + r.result + '" style="max-width:100%;border-radius:8px"><div style="margin-top:6px">';
      const tail = '</div>';
      if (d.error) { $('rec_out').innerHTML = head + '<span class="meta">' + esc(d.error) + '</span>' + tail; return; }
      if (d.certain) {
        $('rec_out').innerHTML = head + '<b style="color:var(--ok)">' + esc(d.verdict) + '</b> <span class="meta">(' + d.matches[0].cos + ')</span><br>' + esc((d.advice || '').slice(0, 400)) + tail;
      } else {
        $('rec_out').innerHTML = head + '<b style="color:var(--bad)">НЕ УВЕРЕН</b><br><span class="meta">не употребляй в пищу</span>' + d.matches.map(m => '<br>похоже: ' + esc(m.species) + ' (' + m.cos + ')').join('') + tail;
      }
    });
  };
  if (false) api('/api/recognize', {b64: '', name: ''}).then(d => {
      if (d.error) { $('rec_out').innerHTML = '<span class="meta">' + esc(d.error) + '</span>'; return; }
      if (d.certain) {
        $('rec_out').innerHTML = '<b style="color:var(--ok)">' + esc(d.verdict) +
          '</b> <span class="meta">(' + d.matches[0].cos + ')</span><br>' +
          esc((d.advice || '').slice(0, 400));
      } else {
        $('rec_out').innerHTML = '<b style="color:var(--bad)">НЕ УВЕРЕН</b>' +
          '<br><span class="meta">не употребляй в пищу</span>' +
          d.matches.map(m => '<br>похоже: ' + esc(m.species) + ' (' + m.cos + ')').join('');
      }
    });
  r.readAsDataURL(f);
});
async function loadSpecies() {
  const d = await api('/api/species');
  $('species').innerHTML = Object.entries(d).map(([s, v]) =>
    '<div class="src">' + (v.toxic ? '☠ ' : '') + esc(s) + ' — ' + v.photos + ' фото</div>'
  ).join('') || '—';
}
loadSpecies();
loadSources();
}
async function fb(sign, idx, el) {
  if (el) { el.disabled = true; el.textContent = sign>0 ? '✓ учтено' : '✖ забыто'; }
  try {
    const r = await api('/api/feedback', {sign, idx});
    if (r.error) { $('stat').textContent = 'ОШИБКА: ' + r.error; return; }
    $('stat').textContent = (sign>0?'подтверждено':'стёрто') + ' · ' + (r.fam!=null?('знакомость '+r.fam.toFixed(2)):'');
  } catch (e) { $('stat').textContent = 'СЕТЬ: ' + e; }
}
async function loadSources() {
  const d = await api('/api/sources');
  $('sources').innerHTML = Object.entries(d).map(([s,n]) =>
    '<div class="src" title="'+esc(s)+'">'+esc(s.split('/').pop())+' — '+n+'</div>').join('') || '—';
}
function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
$('q').addEventListener('keydown', e => { if (e.key === 'Enter') search(); });
$('idx').addEventListener('keydown', e => { if (e.key==='Enter' && e.target.value.trim())
  api('/api/index', {path: e.target.value.trim()}).then(r => { $('stat').textContent =
  'индексировано: ' + r.added + ' эпизодов, ' + r.dups + ' дублей'; loadSources(); });
  e.target.value=''; });
$('mem').addEventListener('keydown', e => { if (e.key==='Enter' && e.target.value.trim())
  api('/api/memorize', {text: e.target.value.trim()}).then(() => $('stat').textContent='запомнил');
  e.target.value=''; });
$('sav').addEventListener('keydown', e => { if (e.key==='Enter' && e.target.value.trim())
  api('/api/save', {path: e.target.value.trim()}).then(() => $('stat').textContent='сохранено');
  e.target.value=''; });
loadSources();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    system = None

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json; charset=utf-8"):
        body = (HTML if ctype.startswith("text/html") else json.dumps(
            obj, ensure_ascii=False)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(None, ctype="text/html; charset=utf-8")
        if u.path == "/api/search":
            q = parse_qs(u.query).get("q", [""])[0]
            sys_ = Handler.system
            res, text = sys_.answer(q)
            ext = None
            if text and res.candidates:
                from answer_extract import detect_type
                if detect_type(q):
                    import answer_extract as ae
                    e = ae.extract_answer(q, [(c.text, sys_.memory.meta[c.idx])
                                              for c in res.candidates[:3]])
                    if e and e["conf"] >= 0.5:
                        ext = e
            return self._send({
                "text": text, "zone": res.zone, "familiarity": res.familiarity,
                "n": len(sys_.memory),
                "extracted": ext,
                "candidates": [
                    {"idx": c.idx, "text": c.text, "conf": c.conf,
                     "citation": sys_.citation(c)} for c in res.candidates],
            })
        if u.path == "/api/sources":
            return self._send(Handler.system.memory.sources())
        if u.path == "/api/species":
            sys_ = Handler.system
            counts: dict = {}
            for m in sys_.memory.meta:
                if m and m.get("kind") == "image":
                    sp = m.get("species", "?")
                    counts[sp] = counts.get(sp, 0) + 1
            return self._send({sp: {"photos": n, **sys_.species_info.get(sp, {})}
                               for sp, n in sorted(counts.items())})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        except Exception:
            return self._send({"error": "bad json"}, 400)
        sys_ = Handler.system
        if u.path == "/api/feedback":
            try:
                from memory import Candidate
                idx = int(body.get("idx", -1))
                cand = None
                if 0 <= idx < len(sys_.memory):
                    cand = Candidate(idx, sys_.memory.texts[idx], 0.0, 0.5)
                st = sys_.feedback(float(body.get("sign", 1)), cand=cand)
                return self._send({"ok": st is not None,
                                   "fam": st["familiarity_after"] if st else None})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._send({"ok": False, "error": str(e)})
        if u.path == "/api/recognize":
            import base64
            import tempfile
            raw = base64.b64decode(body["b64"])
            fd, p = tempfile.mkstemp(suffix=".jpg")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            try:
                out = sys_.recognize(p)
            finally:
                os.unlink(p)
            return self._send(out)
        if u.path == "/api/memorize":
            sys_.memorize(body["text"])
            return self._send({"ok": True, "n": len(sys_.memory)})
        if u.path == "/api/index":
            r = sys_.index_path(body["path"])
            return self._send(r)
        if u.path == "/api/save":
            sys_.save(body["path"])
            return self._send({"ok": True})
        self._send({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None)
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    from config import FlyConfig
    from embedder import HashingEmbedder
    Handler.system = FlySystem(FlyConfig(), HashingEmbedder())
    if args.state and os.path.exists(args.state):
        Handler.system.load(args.state)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"мозг мухи: http://{args.host}:{args.port} "
          f"({len(Handler.system.memory)} эпизодов)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nпока.")


if __name__ == "__main__":
    main()
