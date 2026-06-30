"""Local webcam and image upoad demo for SpotFakePhoto."""

import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, UnidentifiedImageError

from predict import GLOBAL_MODEL_PATH, load_model, predict_image


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpotFakePhoto Camera</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; background:#111315; color:#f4f5f6; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:#111315; }
    header { height:64px; display:flex; align-items:center; justify-content:space-between; padding:0 28px; border-bottom:1px solid #303438; }
    h1 { margin:0; font-size:18px; letter-spacing:0; }
    .status { display:flex; align-items:center; gap:8px; color:#aeb4ba; font-size:13px; }
    .dot { width:9px; height:9px; border-radius:50%; background:#697078; }
    .dot.live { background:#35c878; box-shadow:0 0 0 4px rgba(53,200,120,.12); }
    main { display:grid; grid-template-columns:minmax(0,1fr) 330px; min-height:calc(100vh - 64px); }
    .stage { position:relative; display:grid; place-items:center; background:#090a0b; overflow:hidden; min-height:520px; }
    video, #preview { width:100%; height:100%; object-fit:contain; position:absolute; inset:0; }
    #preview { display:none; }
    .empty { color:#80878e; text-align:center; padding:24px; }
    .empty strong { display:block; color:#d8dcdf; font-size:20px; margin-bottom:8px; }
    aside { padding:28px; border-left:1px solid #303438; background:#171a1d; }
    .label { color:#969da4; font-size:12px; text-transform:uppercase; font-weight:700; }
    .result { margin:10px 0 6px; font-size:30px; font-weight:750; }
    .score { font-variant-numeric:tabular-nums; color:#bec4c9; font-size:15px; }
    .meter { height:8px; background:#30353a; margin:18px 0 28px; overflow:hidden; border-radius:4px; }
    .fill { width:50%; height:100%; background:#7e8790; transition:width .25s, background .25s; }
    .controls { display:grid; gap:10px; }
    button, .upload { min-height:44px; border:1px solid #41474d; background:#24282c; color:#f5f6f7; padding:0 16px; font:600 14px inherit; cursor:pointer; display:flex; align-items:center; justify-content:center; }
    button.primary { background:#f4f5f6; color:#111315; border-color:#f4f5f6; }
    button:hover, .upload:hover { filter:brightness(1.12); }
    button:disabled { opacity:.45; cursor:not-allowed; }
    input { display:none; }
    .note { margin-top:24px; color:#858c93; font-size:12px; line-height:1.5; }
    .error { color:#ff8585; margin-top:14px; font-size:13px; min-height:20px; }
    canvas { display:none; }
    @media (max-width:760px) {
      main { grid-template-columns:1fr; grid-template-rows:minmax(360px,58vh) auto; }
      aside { border-left:0; border-top:1px solid #303438; padding:22px; }
      header { padding:0 18px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>SpotFakePhoto</h1>
    <div class="status"><span id="dot" class="dot"></span><span id="status">Camera idle</span></div>
  </header>
  <main>
    <section class="stage">
      <div id="empty" class="empty"><strong>No camera feed</strong>Start the camera or choose an image.</div>
      <video id="video" autoplay muted playsinline></video>
      <img id="preview" alt="Uploaded image preview">
      <canvas id="canvas"></canvas>
    </section>
    <aside>
      <div class="label">Prediction</div>
      <div id="result" class="result">Waiting</div>
      <div id="score" class="score">Screen probability: --</div>
      <div class="meter"><div id="fill" class="fill"></div></div>
      <div class="controls">
        <button id="start" class="primary">Start camera</button>
        <button id="stop" disabled>Stop camera</button>
        <label class="upload" for="file">Choose image</label>
        <input id="file" type="file" accept="image/*">
      </div>
      <div id="error" class="error"></div>
      <p class="note">Below 50% means original photo. At or above 50% means screen recapture. Scores near 50% are uncertain.</p>
    </aside>
  </main>
  <script>
    const video=document.querySelector('#video'), canvas=document.querySelector('#canvas');
    const preview=document.querySelector('#preview'), empty=document.querySelector('#empty');
    const start=document.querySelector('#start'), stop=document.querySelector('#stop');
    const file=document.querySelector('#file'), result=document.querySelector('#result');
    const score=document.querySelector('#score'), fill=document.querySelector('#fill');
    const status=document.querySelector('#status'), dot=document.querySelector('#dot');
    const error=document.querySelector('#error');
    let stream=null, timer=null, busy=false;

    function showPrediction(p) {
      const uncertain=Math.abs(p-.5)<.08, screen=p>=.5;
      result.textContent=uncertain?'Uncertain':(screen?'Screen recapture':'Original photo');
      score.textContent=`Screen probability: ${(p*100).toFixed(1)}%`;
      fill.style.width=`${p*100}%`;
      fill.style.background=uncertain?'#e2b84b':(screen?'#ee6666':'#35c878');
    }
    async function classify(blob) {
      if (busy) return;
      busy=true;
      try {
        const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob});
        const data=await response.json();
        if (!response.ok) throw new Error(data.error||'Prediction failed');
        showPrediction(data.probability);
        error.textContent='';
      } catch (e) { error.textContent=e.message; }
      finally { busy=false; }
    }
    async function capture() {
      if (!stream || video.videoWidth===0) return;
      canvas.width=video.videoWidth; canvas.height=video.videoHeight;
      canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
      canvas.toBlob(classify,'image/jpeg',.9);
    }
    start.onclick=async()=>{
      try {
        stream=await navigator.mediaDevices.getUserMedia({
          video:{
            facingMode:{ideal:'environment'},
            width:{ideal:1920},
            height:{ideal:1080}
          },
          audio:false
        });
        video.srcObject=stream; video.style.display='block'; preview.style.display='none'; empty.style.display='none';
        start.disabled=true; stop.disabled=false; status.textContent='Live prediction'; dot.classList.add('live');
        timer=setInterval(capture,1500);
      } catch(e) { error.textContent='Camera permission was denied or no camera is available.'; }
    };
    stop.onclick=()=>{
      clearInterval(timer); stream?.getTracks().forEach(track=>track.stop()); stream=null;
      video.srcObject=null; video.style.display='none'; empty.style.display='block';
      start.disabled=false; stop.disabled=true; status.textContent='Camera idle'; dot.classList.remove('live');
    };
    file.onchange=()=>{
      const selected=file.files[0]; if(!selected) return;
      if(stream) stop.click();
      preview.src=URL.createObjectURL(selected); preview.style.display='block'; empty.style.display='none';
      status.textContent='Uploaded image'; classify(selected);
    };
  </script>
</body>
</html>"""


class DemoHandler(BaseHTTPRequestHandler):
    native_model = None
    global_model = None
    inference_lock = threading.Lock()

    def send_bytes(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/":
            self.send_bytes(404, "text/plain; charset=utf-8", b"Not found")
            return
        self.send_bytes(200, "text/html; charset=utf-8", HTML.encode("utf-8"))

    def do_POST(self):
        if self.path != "/predict":
            self.send_bytes(404, "application/json", b'{"error":"Not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 12 * 1024 * 1024:
                raise ValueError("Image must be between 1 byte and 12 MB")
            image = Image.open(io.BytesIO(self.rfile.read(length)))
            with self.inference_lock:
                probability = predict_image(image, self.native_model, self.global_model)
            body = json.dumps({"probability": probability}).encode("utf-8")
            self.send_bytes(200, "application/json", body)
        except (ValueError, UnidentifiedImageError, OSError) as exc:
            body = json.dumps({"error": str(exc) or "Invalid image"}).encode("utf-8")
            self.send_bytes(400, "application/json", body)

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Run the SpotFakePhoto webcam demo.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("Loading models...")
    DemoHandler.native_model = load_model()
    DemoHandler.global_model = load_model(GLOBAL_MODEL_PATH)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    print(f"Camera demo: http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
