# main.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, Optional, List
import httpx
from bs4 import BeautifulSoup
import uuid
import os
import logging
import ipaddress, socket
from urllib.parse import urlparse
import time
import asyncio

# load .env
try:
    from dotenv import load_dotenv  # pip install python-dotenv

    load_dotenv()
except Exception:
    pass

# OpenAI
from openai import OpenAI

_openai_client: Optional[OpenAI] = None
_last_key: Optional[str] = None


def _get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=400,
                            detail="OpenAI API key not set. Define OPENAI_API_KEY in your environment/.env.")
    global _openai_client, _last_key
    if _openai_client is None or _last_key != key:
        _openai_client = OpenAI(api_key=key)
        _last_key = key
    return _openai_client


# ------------------------------
# Setup
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blog-app")

app = FastAPI(
    title="Blog → Summary → Podcast",
    description="Scrape with Firecrawl, summarize with OpenAI GPT-4/4o, and generate podcast audio with ElevenLabs.",
    version="2.0.1"
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ------------------------------
# Models & in-memory store
# ------------------------------
class BlogStatus(BaseModel):
    id: str
    status: str  # processing | completed | failed
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    word_count: Optional[int] = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())
    # summary
    summary: Optional[str] = None
    summary_style: Optional[str] = None
    # podcast
    podcast_path: Optional[str] = None
    podcast_duration_sec: Optional[float] = None


blog_extracts: Dict[str, BlogStatus] = {}


class BlogRequest(BaseModel):
    blog_url: HttpUrl


class SummarizeRequest(BaseModel):
    style: Optional[str] = "bullet"  # bullet | paragraph | executive


# ------------------------------
# Helpers
# ------------------------------
def _is_safe_url(raw_url: str) -> bool:
    """Basic SSRF guard. Blocks internal/private addresses."""
    try:
        parsed = urlparse(raw_url)
        if parsed.scheme not in ("http", "https"):
            return False
        infos = socket.getaddrinfo(parsed.hostname, None)
        ips = {ai[4][0] for ai in infos}
        for ip in ips:
            ip_obj = ipaddress.ip_address(ip)
            if (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
                    or ip_obj.is_reserved or ip_obj.is_multicast):
                return False
        return True
    except Exception:
        return False


def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    if not text:
        return []
    chunks, start = [], 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        cut = text.rfind("\n\n", start, end)
        if cut == -1 or cut <= start + 1500:
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return chunks


async def _summarize_chunk(chunk: str, purpose: str) -> str:
    resp = _get_openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a precise summarizer. Keep facts accurate and avoid speculation."},
            {"role": "user", "content": f"{purpose}\n\nText:\n{chunk}"}
        ]
    )
    return resp.choices[0].message.content.strip()


async def _summarize_multi(chunks: List[str], title: str = "", style: str = "bullet") -> str:
    partials = []
    for i, c in enumerate(chunks, 1):
        part = await _summarize_chunk(
            c,
            purpose=f"Summarize part {i}/{len(chunks)} of an article{' titled: ' + title if title else ''}. "
                    f"Output 5–8 crisp bullets with key facts, names, numbers, and dates."
        )
        partials.append(part)
    joined = "\n\n".join(f"Part {i}:\n{p}" for i, p in enumerate(partials, 1))
    style_prompt = {
        "bullet": "Return 6–10 punchy bullets, grouped by theme with short subheaders.",
        "paragraph": "Return a 1–3 paragraph narrative summary (120–220 words).",
        "executive": "Return an executive brief: 3 bullets 'What it is', 3 'Why it matters', 3 'Key numbers/dates'."
    }.get(style, "Return 6–10 punchy bullets.")
    final = _get_openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": "You are an expert editor who synthesizes multiple notes into one accurate summary."},
            {"role": "user",
             "content": f"Combine the part-summaries into ONE final summary. {style_prompt}\n\n{joined}"}
        ]
    ).choices[0].message.content.strip()
    if len(final) > 2000:
        final = final[:1997] + "..."
    return final


async def _firecrawl_fetch(url: str, api_key: str) -> Optional[str]:
    """Use Firecrawl to get full text content."""
    fc_endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"url": str(url)}  # ensure plain str
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(fc_endpoint, headers=headers, json=payload)
        if r.status_code == 200:
            data = r.json()
            text = data.get("content") or data.get("markdown") or data.get("text") or ""
            return text.strip()
        logger.warning(f"Firecrawl error {r.status_code}: {r.text}")
        return None


def _bs4_fallback(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'menu']):
        tag.decompose()
    candidates = [
        "article", '[role="main"]', ".post-content", ".entry-content", ".content",
        ".article-content", ".post-body", ".article-body", "main", ".single-post-content"
    ]
    element = None
    for sel in candidates:
        found = soup.select_one(sel)
        if found:
            element = found
            break
    if not element:
        element = soup.find("body") or soup
    text = element.get_text(separator="\n\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n\n".join([ln for ln in lines if ln])


async def _elevenlabs_tts(text: str, api_key: str, voice_id: str, out_path: str) -> Optional[float]:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return None
        logger.error(f"ElevenLabs error {r.status_code}: {r.text}")
        return None


# ------------------------------
# Scraper
# ------------------------------
class BlogScraper:
    @staticmethod
    async def extract_content(url: str) -> dict:
        url = str(url)  # ensure plain string for httpx
        logger.info(f"Scraping URL via Firecrawl (if configured): {url}")
        fc_key = os.getenv("FIRECRAWL_API_KEY")
        if fc_key:
            try:
                text = await _firecrawl_fetch(url, fc_key)
                if text and text.strip():
                    return {"success": True, "title": None, "content": text}
            except Exception as e:
                logger.warning(f"Firecrawl failed, falling back. {e}")

        # Fallback: direct GET + BeautifulSoup
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=httpx.Timeout(30.0)) as client:
            for attempt in range(3):
                try:
                    r = await client.get(url)  # url is str
                    r.raise_for_status()
                    html = r.content
                    soup = BeautifulSoup(html, "html.parser")
                    title_text = "Article"
                    h1 = soup.find("h1")
                    title_tag = soup.find("title")
                    if h1 and h1.get_text(strip=True):
                        title_text = h1.get_text(strip=True)
                    elif title_tag:
                        title_text = title_tag.get_text(strip=True)
                    content_text = _bs4_fallback(html)
                    return {"success": True, "title": title_text, "content": content_text}
                except Exception as e:
                    if attempt == 2:
                        return {"success": False, "error": f"Fetch failed: {e}"}
                    continue


# ------------------------------
# Tasks
# ------------------------------
async def process_blog_extraction(extract_id: str, request: BlogRequest):
    try:
        blog_extracts[extract_id].status = "processing"
        # cast HttpUrl -> str here too
        result = await BlogScraper.extract_content(str(request.blog_url))
        if result.get("success"):
            content = (result.get("content") or "").strip()
            title = result.get("title") or "Article"
            # normalize spacing
            content = "\n\n".join([p.strip() for p in content.splitlines() if p.strip()])
            blog_extracts[extract_id].status = "completed"
            blog_extracts[extract_id].title = title
            blog_extracts[extract_id].content = content
            blog_extracts[extract_id].word_count = len(content.split())
        else:
            blog_extracts[extract_id].status = "failed"
            blog_extracts[extract_id].error = result.get("error", "Extraction failed")
    except Exception as e:
        logger.exception(f"Extraction failed: {e}")
        blog_extracts[extract_id].status = "failed"
        blog_extracts[extract_id].error = f"Processing failed: {e}"


# ------------------------------
# UI (no sidebar)
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Blog → Summary → Podcast</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b1220}
  .wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
  .card{max-width:980px;width:100%;background:#fff;border-radius:16px;padding:24px;box-shadow:0 20px 40px rgba(0,0,0,.25)}
  h1{font-size:28px;color:#111827}
  p.muted{color:#6b7280;margin-top:6px}
  .row{display:flex;gap:12px;margin-top:20px}
  input.url{flex:1;padding:14px;border:2px solid #e5e7eb;border-radius:12px;font-size:16px}
  button.primary{padding:14px 18px;border:0;border-radius:12px;background:linear-gradient(45deg,#4f46e5,#06b6d4);color:#fff;font-weight:700;cursor:pointer}
  .status{margin-top:16px;padding:14px;border-radius:12px;display:none}
  .status.processing{background:#fff7ed;border:1px solid #fed7aa;color:#7c2d12;display:block}
  .status.completed{background:#ecfeff;border:1px solid #67e8f9;color:#164e63;display:block}
  .status.failed{background:#fee2e2;border:1px solid #fecaca;color:#7f1d1d;display:block}
  .content-result{margin-top:16px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:16px}
  .meta{background:#eef2ff;border:1px solid #c7d2fe;padding:10px;border-radius:10px;margin-bottom:12px;color:#1e3a8a}
  .content-text{white-space:pre-wrap;line-height:1.6;background:#fff;border:1px solid #e5e7eb;padding:14px;border-radius:10px}
  .btn{padding:10px 14px;border:0;border-radius:8px;background:#10b981;color:#fff;font-weight:600;cursor:pointer}
  .btn.purple{background:#6d28d9}
  .btn.indigo{background:#4f46e5}
  .flex{display:flex;gap:10px;flex-wrap:wrap}
  .summary-box{margin-top:12px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px;white-space:pre-wrap;line-height:1.6}
  .small{font-size:12px;color:#6b7280;margin-top:6px}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Blog → Summary → Podcast</h1>
    <p class="muted">Keys are read from your environment (.env). Paste a public blog URL and go.</p>
    <div class="row">
      <input id="blog_url" class="url" type="url" placeholder="https://example.com/blog-post" required>
      <button id="extractBtn" class="primary">Extract</button>
    </div>
    <div id="status" class="status"><span id="statusMessage"></span></div>
  </div>
</div>

<script>
  const statusEl = document.getElementById('status');
  const statusMsg = document.getElementById('statusMessage');
  const extractBtn = document.getElementById('extractBtn');
  let pollInterval;

  function setStatus(cls, msg){
    statusEl.className = 'status ' + cls;
    statusEl.style.display = 'block';
    statusMsg.textContent = msg;
  }

  function escapeHtml(s=''){
    return s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }

  function renderResult(extraction){
    statusEl.className = 'status completed';
    statusMsg.innerHTML = '';
    const wrap = document.createElement('div'); wrap.className = 'content-result';

    const h = document.createElement('h3'); h.textContent = `${extraction.title || 'Article'}`; wrap.appendChild(h);

    const meta = document.createElement('div'); meta.className='meta';
    meta.innerHTML = `<div><strong>URL:</strong> ${escapeHtml(extraction.url)}</div>
                      <div><strong>Word Count:</strong> ${Number(extraction.word_count) || 0} words</div>
                      <div><strong>Characters:</strong> ${(extraction.content || '').length} chars</div>`;
    wrap.appendChild(meta);

    const pre = document.createElement('pre'); pre.className='content-text'; pre.textContent = extraction.content || ''; wrap.appendChild(pre);

    const btns = document.createElement('div'); btns.className='flex';
    const copyBtn = document.createElement('button'); copyBtn.className='btn'; copyBtn.textContent='Copy Content';
    copyBtn.onclick = () => navigator.clipboard.writeText(extraction.content || ''); btns.appendChild(copyBtn);

    const sumBtn = document.createElement('button'); sumBtn.className='btn purple'; sumBtn.textContent='Summarize (GPT-4)';
    sumBtn.onclick = () => summarize(extraction.id, 'bullet'); btns.appendChild(sumBtn);

    const podBtn = document.createElement('button'); podBtn.className='btn indigo'; podBtn.textContent='Generate Podcast';
    podBtn.onclick = () => podcast(extraction.id); btns.appendChild(podBtn);

    wrap.appendChild(btns);

    const summaryBox = document.createElement('div'); summaryBox.id = `summary-${extraction.id}`; summaryBox.className='summary-box'; summaryBox.style.display='none'; wrap.appendChild(summaryBox);
    const audioBox = document.createElement('div'); audioBox.id = `audio-${extraction.id}`; audioBox.className='summary-box'; audioBox.style.display='none'; wrap.appendChild(audioBox);

    statusMsg.appendChild(wrap);
  }

  document.getElementById('extractBtn').addEventListener('click', async () => {
    const url = document.getElementById('blog_url').value.trim();
    if(!url){ alert('Enter a URL'); return; }
    extractBtn.disabled = true;
    setStatus('processing', 'Starting extraction…');
    try{
      const r = await fetch('/extract',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ blog_url: url })});
      const j = await r.json();
      if(!r.ok) throw new Error(j.detail || 'Failed to start extraction');
      setStatus('processing', 'Fetching & parsing…');
      pollExtractionStatus(j.id);
    }catch(e){
      setStatus('failed', `${e.message}`);
      extractBtn.disabled = false;
    }
  });

  async function pollExtractionStatus(id){
    clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
      try{
        const r = await fetch(`/status/${id}`); const j = await r.json();
        if(j.status === 'completed'){
          clearInterval(pollInterval);
          renderResult(j);
          extractBtn.disabled = false;
          extractBtn.textContent = 'Extract Another';
        } else if (j.status === 'failed'){
          clearInterval(pollInterval);
          setStatus('failed', `${j.error || 'Extraction failed'}`);
          extractBtn.disabled = false;
        } else {
          setStatus('processing', 'Extracting content…');
        }
      }catch(e){
        clearInterval(pollInterval);
        setStatus('failed', `${e.message}`);
        extractBtn.disabled = false;
      }
    }, 1000);
  }

  async function summarize(id, style='bullet'){
    const box = document.getElementById(`summary-${id}`);
    box.style.display='block';
    box.textContent = 'Summarizing with GPT-4… (≤ 2000 chars)';
    try{
      const r = await fetch(`/summarize/${id}`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({style})});
      const j = await r.json();
      if(!r.ok) throw new Error(j.detail || 'Summarization failed');
      box.textContent = j.summary || '(No summary)';
    }catch(e){
      box.textContent = `${e.message}`;
    }
  }

  async function podcast(id){
    const box = document.getElementById(`audio-${id}`);
    box.style.display='block';
    box.textContent = 'Generating podcast audio…';
    try{
      const r = await fetch(`/podcast/${id}`, {method:'POST'});
      const j = await r.json();
      if(!r.ok) throw new Error(j.detail || 'Audio failed');
      box.innerHTML = `<div>Podcast ready:</div>
        <audio controls src="${j.podcast_url}" style="margin-top:8px;width:100%"></audio>
        <div class="small" style="margin-top:8px"><a href="${j.podcast_url}" download>Download MP3</a></div>`;
    }catch(e){
      box.textContent = `${e.message}`;
    }
  }
</script>
</body>
</html>
    """


# ------------------------------
# API routes
# ------------------------------
@app.post("/extract")
async def start_content_extraction(request: BlogRequest):
    if not _is_safe_url(str(request.blog_url)):
        raise HTTPException(status_code=400, detail="URL not allowed.")
    extract_id = str(uuid.uuid4())
    blog_extracts[extract_id] = BlogStatus(id=extract_id, status="processing", url=str(request.blog_url))
    asyncio.create_task(process_blog_extraction(extract_id, request))
    return {"id": extract_id, "status": "started"}


@app.get("/status/{extract_id}")
async def get_extraction_status(extract_id: str):
    if extract_id not in blog_extracts:
        raise HTTPException(status_code=404, detail="Extraction not found")
    return blog_extracts[extract_id]


@app.get("/extractions")
async def list_all_extractions():
    return list(blog_extracts.values())


@app.delete("/extractions/{extract_id}")
async def delete_extraction(extract_id: str):
    if extract_id not in blog_extracts:
        raise HTTPException(status_code=404, detail="Extraction not found")
    path = blog_extracts[extract_id].podcast_path
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    del blog_extracts[extract_id]
    return {"message": "Extraction deleted successfully"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/summarize/{extract_id}")
async def summarize_extraction(extract_id: str, req: SummarizeRequest = Body(default=SummarizeRequest())):
    if extract_id not in blog_extracts:
        raise HTTPException(status_code=404, detail="Extraction not found")
    rec = blog_extracts[extract_id]
    if rec.status != "completed" or not rec.content:
        raise HTTPException(status_code=400, detail="Content not available for summarization")
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OpenAI API key not set (OPENAI_API_KEY).")
    try:
        chunks = _chunk_text(rec.content, max_chars=8000)
        final_summary = await _summarize_multi(chunks, title=rec.title or "", style=req.style or "bullet")
        rec.summary = final_summary
        rec.summary_style = req.style
        return {"id": extract_id, "summary": final_summary, "style": req.style}
    except Exception as e:
        logger.exception(f"Summarization failed for {extract_id}: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed")


@app.post("/podcast/{extract_id}")
async def generate_podcast(extract_id: str):
    if extract_id not in blog_extracts:
        raise HTTPException(status_code=404, detail="Extraction not found")
    rec = blog_extracts[extract_id]
    if not rec.summary:
        raise HTTPException(status_code=400, detail="No summary found. Run summarization first.")
    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    if not eleven_key:
        raise HTTPException(status_code=400, detail="ElevenLabs API key not set (ELEVENLABS_API_KEY).")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    fname = f"podcast_{extract_id}.mp3"
    out_path = os.path.join("static", fname)
    try:
        dur = await _elevenlabs_tts(rec.summary, eleven_key, voice_id, out_path)
        rec.podcast_path = out_path
        rec.podcast_duration_sec = dur
        return {"id": extract_id, "podcast_url": f"/static/{fname}", "approx_duration_sec": dur}
    except Exception as e:
        logger.exception(f"TTS failed for {extract_id}: {e}")
        raise HTTPException(status_code=500, detail="Podcast generation failed")


@app.get("/test-url")
async def test_url_accessibility(url: str):
    if not _is_safe_url(url):
        return {"url": url, "accessible": False, "error": "URL not allowed"}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.head(str(url), headers=headers, follow_redirects=True)
        return {
            "url": url,
            "accessible": True,
            "status_code": r.status_code,
            "content_type": r.headers.get('content-type', 'Unknown'),
            "server": r.headers.get('server', 'Unknown')
        }
    except Exception as e:
        return {"url": url, "accessible": False, "error": str(e)}


# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
