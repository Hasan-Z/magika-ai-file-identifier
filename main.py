import os
import time
import subprocess
import sys
import warnings
import asyncio
import importlib.metadata
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

# Suppress TensorFlow logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = FastAPI(title="Magika AI Identifier")

# State Management
magika_instance = None
model_ready = False
magika_version = "Unknown"

async def load_model_in_background():
    global magika_instance, model_ready, magika_version
    try:
        from magika import Magika
        magika_instance = Magika()
        try:
            magika_version = importlib.metadata.version("magika")
        except:
            magika_version = "1.0.2"
        model_ready = True
    except Exception as e:
        print(f"--- [SYSTEM] Initialization Error: {e} ---")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_model_in_background())

def extract_magika_info(res):
    final_score = 0.0
    final_label = "UNKNOWN"
    raw_dict = {}
    try:
        raw_dict = {
            "label": getattr(res, "label", "N/A"),
            "score": float(getattr(res, "score", 0.0)),
            "dl": {
                "ct_label": getattr(res.dl, "ct_label", "N/A") if hasattr(res, 'dl') else "N/A",
                "score": float(getattr(res.dl, "score", 0.0)) if hasattr(res, 'dl') else 0.0
            } if hasattr(res, 'dl') else "N/A"
        }
    except:
        raw_dict = {"status": "Could not serialize"}

    if hasattr(res, 'score'): final_score = res.score
    elif hasattr(res, 'dl') and hasattr(res.dl, 'score'): final_score = res.dl.score
    if hasattr(res, 'dl') and hasattr(res.dl, 'ct_label'): final_label = res.dl.ct_label
    elif hasattr(res, 'output') and hasattr(res.output, 'ct_label'): final_label = res.output.ct_label
    elif hasattr(res, 'label'): final_label = res.label
    return final_label.upper(), float(final_score), raw_dict

@app.get("/status")
async def get_status():
    return {"ready": model_ready, "version": magika_version}

@app.post("/predict_slices")
async def predict_slices(head: UploadFile = File(...), mid: UploadFile = File(...), tail: UploadFile = File(...)):
    if not model_ready: raise HTTPException(status_code=503, detail="Initializing")
    try:
        h_bytes, m_bytes, t_bytes = await head.read(), await mid.read(), await tail.read()
        start_time = time.perf_counter()
        res = magika_instance.identify_bytes(h_bytes + m_bytes + t_bytes)
        latency_ms = (time.perf_counter() - start_time) * 1000
        label, score, raw = extract_magika_info(res)
        return {"label": label, "confidence": f"{score:.2%}", "latency_ms": f"{latency_ms:.2f}ms", "raw": raw}
    except Exception as e:
        return {"label": "ERROR", "confidence": "0%", "latency_ms": str(e), "raw": {}}

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <title>Magika AI | Identifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <script>
        tailwind.config = { darkMode: 'class', theme: { extend: { colors: { appleBlue: '#0A84FF', appleBlueDark: '#0056b3' } } } }
    </script>
    <style>
        body { font-family: 'Inter', sans-serif; transition: background 0.5s ease; min-height: 100vh; }
        .dark body { background: radial-gradient(circle at top left, #1c1c1e, #000); color: white; }
        .light body { background: radial-gradient(circle at top left, #f5f5f7, #ffffff); color: #1d1d1f; }
        .glass { backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px); border: 1px solid rgba(150,150,150,0.1); }
        .dark .glass { background: rgba(255, 255, 255, 0.03); box-shadow: 0 30px 60px rgba(0,0,0,0.5); }
        .light .glass { background: rgba(255, 255, 255, 0.7); box-shadow: 0 30px 60px rgba(0,0,0,0.1); }
        .code-bg { background: rgba(0,0,0,0.3); font-family: 'Courier New', monospace; }
        .rotate-180 { transform: rotate(180deg); }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb { background: rgba(10, 132, 255, 0.2); border-radius: 10px; }
    </style>
</head>
<body class="flex flex-col items-center p-6 py-20">

    <div class="fixed top-10 right-10 z-50">
        <button onclick="toggleTheme()" class="glass p-3 rounded-full hover:scale-110 active:scale-95 transition-all">
            <i id="theme-icon" data-lucide="moon" class="w-5 h-5"></i>
        </button>
    </div>

    <div class="max-w-xl w-full mb-12">
        <div class="text-center mb-10">
            <h1 class="text-4xl font-semibold tracking-tight">Magika</h1>
            <p class="text-gray-500 uppercase text-[10px] tracking-[0.4em] mt-2 font-bold mb-4">AI File Identification</p>
            
            <div id="ver-badge" class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-appleBlue/10 border border-appleBlue/20 opacity-0 transition-opacity duration-1000">
                <span class="w-1.5 h-1.5 rounded-full bg-appleBlue animate-pulse"></span>
                <span class="text-[9px] font-mono font-bold text-appleBlue uppercase tracking-widest">Google Magika v<span id="ver-num">...</span></span>
            </div>
        </div>

        <div id="card" class="glass rounded-[3rem] p-10 transition-all duration-500 relative">
            <div id="init-zone" class="py-10 text-center">
                <div class="w-12 h-12 border-4 border-appleBlue/20 border-t-appleBlue rounded-full animate-spin mx-auto mb-6"></div>
                <h3 class="text-xl font-medium mb-2">System Initializing</h3>
                <p class="text-sm text-gray-500 text-xs">Loading Deep Learning weights...</p>
            </div>

            <div id="upload-zone" class="hidden">
                <div id="drop-zone" class="border-2 border-dashed border-gray-400/20 rounded-[2.5rem] p-16 text-center cursor-pointer hover:border-appleBlue/50 transition-all group">
                    <div class="w-16 h-16 bg-appleBlue/10 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                        <i data-lucide="shield-check" class="w-8 h-8 text-appleBlue"></i>
                    </div>
                    <p class="text-lg font-medium">Identify File</p>
                    <p class="text-xs text-gray-500 mt-2">Instant Deep Learning Scan</p>
                    <input type="file" id="fileInput" class="hidden">
                </div>
            </div>

            <div id="loading" class="hidden py-10 text-center">
                <div class="w-10 h-10 border-2 border-appleBlue border-t-transparent rounded-full animate-spin mx-auto"></div>
                <p class="mt-4 text-gray-500 text-xs uppercase tracking-widest animate-pulse">Analyzing Binary Slices...</p>
            </div>

            <div id="result" class="hidden space-y-8 animate-in fade-in duration-700">
                <div class="flex items-center gap-6">
                    <div id="icon-box" class="w-20 h-20 bg-appleBlue/10 rounded-3xl flex items-center justify-center border border-appleBlue/20">
                        <i id="file-icon" data-lucide="file" class="w-10 h-10 text-appleBlue"></i>
                    </div>
                    <div class="flex-1">
                        <p class="text-[10px] text-gray-500 uppercase tracking-widest mb-1 font-bold">Detection Result</p>
                        <h2 id="res-label" class="text-5xl font-black tracking-tighter uppercase dark:text-white text-gray-900">-</h2>
                    </div>
                    <div class="text-right">
                        <p id="res-score" class="dark:text-appleBlue text-appleBlueDark font-bold text-lg font-mono">-</p>
                        <p class="text-[10px] text-gray-500 uppercase">Confidence</p>
                    </div>
                </div>
                <div class="bg-gray-500/5 rounded-2xl p-4 flex justify-between items-center text-xs">
                    <span class="text-gray-500 uppercase tracking-widest font-semibold">Latency</span>
                    <span id="res-time" class="font-mono dark:text-appleBlue text-appleBlueDark font-bold">-</span>
                </div>
                <button onclick="resetUI()" class="w-full py-5 bg-appleBlue text-white rounded-2xl font-bold text-sm hover:brightness-110 transition-all">Identify New File</button>
            </div>
        </div>
    </div>

    <div id="raw-container" class="max-w-xl w-full hidden opacity-60 hover:opacity-100 transition-opacity pb-20">
        <button onclick="toggleRaw()" class="flex items-center gap-2 text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-4 ml-6">
            <i id="raw-arrow" data-lucide="chevron-down" class="w-3 h-3 transition-transform"></i>
            Full Model Response (JSON)
        </button>
        <div id="raw-content" class="hidden glass rounded-3xl p-6 overflow-hidden">
            <pre id="json-box" class="text-[11px] code-bg p-4 rounded-xl max-h-96 overflow-y-auto whitespace-pre-wrap font-mono"></pre>
        </div>
    </div>

    <script>
        lucide.createIcons();
        function toggleTheme() {
            const h = document.documentElement; const i = document.getElementById('theme-icon');
            if(h.classList.contains('dark')) { h.classList.replace('dark', 'light'); i.setAttribute('data-lucide', 'sun'); }
            else { h.classList.replace('light', 'dark'); i.setAttribute('data-lucide', 'moon'); }
            lucide.createIcons();
        }
        function toggleRaw() {
            document.getElementById('raw-content').classList.toggle('hidden');
            document.getElementById('raw-arrow').classList.toggle('rotate-180');
        }
        async function checkStatus() {
            try {
                const r = await fetch('/status'); const d = await r.json();
                if(d.ready) {
                    document.getElementById('init-zone').classList.add('hidden');
                    document.getElementById('upload-zone').classList.remove('hidden');
                    document.getElementById('ver-badge').classList.remove('opacity-0');
                    document.getElementById('ver-num').innerText = d.version;
                } else { setTimeout(checkStatus, 1500); }
            } catch(e) { setTimeout(checkStatus, 1500); }
        }
        checkStatus();

        const fi = document.getElementById('fileInput');
        document.getElementById('drop-zone').onclick = () => fi.click();
        fi.onchange = async (e) => {
            const f = e.target.files[0]; if(!f) return;
            document.getElementById('upload-zone').classList.add('hidden');
            document.getElementById('loading').classList.remove('hidden');
            const fd = new FormData(); const s = 4096;
            fd.append('head', f.slice(0, s));
            fd.append('mid', f.slice(Math.floor(f.size/2), Math.floor(f.size/2)+s));
            fd.append('tail', f.slice(f.size - s, f.size));
            try {
                const r = await fetch('/predict_slices', { method: 'POST', body: fd });
                const res = await r.json();
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('result').classList.remove('hidden');
                document.getElementById('raw-container').classList.remove('hidden');
                document.getElementById('res-label').innerText = res.label;
                document.getElementById('res-score').innerText = res.confidence;
                document.getElementById('res-time').innerText = res.latency_ms;
                document.getElementById('json-box').innerText = JSON.stringify(res.raw, null, 2);
                lucide.createIcons();
            } catch(e) { alert("Error during analysis."); resetUI(); }
        };
        function resetUI() {
            document.getElementById('result').classList.add('hidden');
            document.getElementById('upload-zone').classList.remove('hidden');
            document.getElementById('raw-container').classList.add('hidden');
            fi.value = "";
        }
    </script>
</body>
</html>
    """