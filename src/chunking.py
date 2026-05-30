def fixed_chunking(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]
        start = 0
        chunk_index = 1

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                    "page": page_number,
                    "text": chunk_text
                })

            start += chunk_size - overlap
            chunk_index += 1

    return chunks


def recursive_chunking(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page in pages:
        paragraphs = page["text"].split("\n")
        current_chunk = ""
        chunk_index = 1

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n"
            else:
                chunks.append({
                    "chunk_id": f"page_{page['page']}_chunk_{chunk_index}",
                    "page": page["page"],
                    "text": current_chunk.strip()
                })

                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n" + paragraph + "\n"
                chunk_index += 1

        if current_chunk.strip():
            chunks.append({
                "chunk_id": f"page_{page['page']}_chunk_{chunk_index}",
                "page": page["page"],
                "text": current_chunk.strip()
            })

    return chunks
