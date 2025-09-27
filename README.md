# Blog → Summary → Podcast

A FastAPI application that:
1. **Scrapes blog articles** (via [Firecrawl](https://firecrawl.dev) API or fallback with BeautifulSoup).  
2. **Summarizes content** using OpenAI GPT models.  
3. **Generates podcast audio** from the summary with ElevenLabs TTS.  

---

## Features
- Extracts and cleans blog content from a public URL  
- Summarizes articles in bullet, paragraph, or executive style  
- Converts summaries into downloadable MP3 podcasts  
- Web UI included (no frontend build needed)  

---

## Requirements

- Python **3.9+**
- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- [httpx](https://www.python-httpx.org/) and [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)
- OpenAI account (API key)
- ElevenLabs account (API key & voice ID)
- (Optional) Firecrawl API key for better scraping

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/blog-summary-podcast.git
cd blog-summary-podcast

# create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows

# install dependencies
pip install -r requirements.txt
