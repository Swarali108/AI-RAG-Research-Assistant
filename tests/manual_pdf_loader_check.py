from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.pdf_loader import load_pdf


print("PDF loader test started...")

pdf_path = project_root / "docs" / "sample.pdf"
pdf_pages = load_pdf(pdf_path)

print(f"Total pages extracted: {len(pdf_pages)}")

for page in pdf_pages[:2]:
    print("Page:", page["page"])
    print(page["text"][:500])
    print("-" * 50)

print("PDF loader test finished.")