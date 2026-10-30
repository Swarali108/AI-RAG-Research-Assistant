from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.pdf_loader import load_pdf
from src.chunking import fixed_chunking, recursive_chunking


pdf_path = project_root / "docs" / "sample.pdf"
pages = load_pdf(pdf_path)

fixed_chunks = fixed_chunking(pages)
recursive_chunks = recursive_chunking(pages)

print(f"Total pages: {len(pages)}")
print(f"Fixed chunks: {len(fixed_chunks)}")
print(f"Recursive chunks: {len(recursive_chunks)}")

print("\nSample fixed chunk:")
print(fixed_chunks[0])

print("\nSample recursive chunk:")
print(recursive_chunks[0])
