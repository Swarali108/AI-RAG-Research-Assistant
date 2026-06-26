"""Multi-format ingestion (#8).

Every loader returns the same page shape the rest of the pipeline expects:
``[{"source": str, "page": int, "text": str}, ...]`` with line structure
preserved so heading-aware chunking can find section boundaries.

Heavy/optional dependencies (python-docx, youtube-transcript-api, pytesseract)
are imported lazily inside their loaders, so the core app runs without them and
each format degrades to a clear error message instead of crashing the service.
"""

import io
import re
from html.parser import HTMLParser
from typing import Any, Dict, List


SUPPORTED_UPLOAD_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".markdown",
                               ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")


def _normalize(text: str) -> str:
    """Collapse intra-line whitespace but keep line breaks (for headings)."""
    lines = [" ".join(line.split()) for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _page(source: str, text: str, page: int = 1) -> Dict[str, Any]:
    return {"source": source, "page": page, "text": _normalize(text)}


def _pdf(data: bytes, source: str) -> List[Dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = _normalize(page.extract_text() or "")
        if text.strip():
            pages.append({"source": source, "page": number, "text": text})
    return pages


def _docx(data: bytes, source: str) -> List[Dict[str, Any]]:
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise RuntimeError("DOCX support needs python-docx (pip install python-docx).") from exc

    document = docx.Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs)
    page = _page(source, text)
    return [page] if page["text"].strip() else []


def _plain(data: bytes, source: str) -> List[Dict[str, Any]]:
    text = data.decode("utf-8", errors="replace")
    page = _page(source, text)
    return [page] if page["text"].strip() else []


def _markdown(data: bytes, source: str) -> List[Dict[str, Any]]:
    # Markdown headings (#, ##) are already heading-aware-friendly; keep as text.
    return _plain(data, source)


def _image_ocr(data: bytes, source: str) -> List[Dict[str, Any]]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Image OCR needs pillow + pytesseract and the Tesseract binary installed."
        ) from exc

    text = pytesseract.image_to_string(Image.open(io.BytesIO(data)))
    page = _page(source, text)
    if not page["text"].strip():
        raise RuntimeError(f"No text could be read from image {source}.")
    return [page]


def extract_pages(filename: str, data: bytes) -> List[Dict[str, Any]]:
    """Dispatch an uploaded file to the right loader by extension."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _pdf(data, filename)
    if name.endswith(".docx"):
        return _docx(data, filename)
    if name.endswith(".txt"):
        return _plain(data, filename)
    if name.endswith((".md", ".markdown")):
        return _markdown(data, filename)
    if name.endswith(_IMAGE_EXTENSIONS):
        return _image_ocr(data, filename)
    raise ValueError(f"Unsupported file type: {filename}")


class _TextExtractor(HTMLParser):
    """Minimal HTML→text: drop script/style, keep visible text and block breaks."""

    _SKIP = {"script", "style", "noscript", "head"}
    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Extract readable text from an HTML string (stdlib only)."""
    parser = _TextExtractor()
    parser.feed(html)
    return _normalize("".join(parser.parts))


def extract_url(url: str, timeout: int = 15, max_chars: int = 200_000) -> List[Dict[str, Any]]:
    """Fetch a web page and extract readable text (stdlib only)."""
    import urllib.request

    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError("URL must start with http:// or https://")

    request = urllib.request.Request(url, headers={"User-Agent": "AI-RAG-Research-Assistant/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (user-provided URL)
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read(max_chars * 4).decode(charset, errors="replace")

    text = html_to_text(html)[:max_chars]
    if not text.strip():
        raise RuntimeError(f"No readable text found at {url}.")
    return [{"source": url, "page": 1, "text": text}]


def _youtube_id(url: str) -> str:
    patterns = [r"v=([A-Za-z0-9_-]{11})", r"youtu\.be/([A-Za-z0-9_-]{11})", r"embed/([A-Za-z0-9_-]{11})"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError("Could not find a YouTube video id in the provided URL.")


def extract_youtube(url: str) -> List[Dict[str, Any]]:
    """Fetch a YouTube transcript and return it as a single page."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError(
            "YouTube support needs youtube-transcript-api (pip install youtube-transcript-api)."
        ) from exc

    video_id = _youtube_id(url)

    # API changed across versions: v1.0+ is instance-based (.fetch), older
    # versions exposed a static .get_transcript. Support both.
    api = YouTubeTranscriptApi()
    if hasattr(api, "fetch"):
        fetched = api.fetch(video_id)
        texts = [getattr(snippet, "text", "") for snippet in fetched]
    elif hasattr(YouTubeTranscriptApi, "get_transcript"):
        texts = [item["text"] for item in YouTubeTranscriptApi.get_transcript(video_id)]
    else:
        raise RuntimeError("Unsupported youtube-transcript-api version.")

    text = _normalize(" ".join(texts))
    if not text.strip():
        raise RuntimeError("Transcript was empty.")
    return [{"source": f"youtube:{video_id}", "page": 1, "text": text}]
