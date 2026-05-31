from pathlib import Path
from pypdf import PdfReader


def load_pdf(file, source_name=None):
    reader = PdfReader(file)

    source = source_name or getattr(file, "name", None)

    if source is None and isinstance(file, (str, Path)):
        source = Path(file).name

    source = source or "uploaded_document.pdf"

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "source": source,
                "page": page_number,
                "text": text.strip(),
            })

    return pages
