import re


def _chunk_metadata(page, chunk_index, chunk_text, method):
    source = page.get("source", "uploaded_document.pdf")
    page_number = page["page"]

    return {
        "chunk_id": f"{source}_page_{page_number}_chunk_{chunk_index}",
        "source": source,
        "page": page_number,
        "method": method,
        "text": chunk_text.strip()
    }


def fixed_chunking(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0
        chunk_index = 1

        while start < len(text):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append(_chunk_metadata(page, chunk_index, chunk_text, "fixed"))

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
                if current_chunk.strip():
                    chunks.append(_chunk_metadata(page, chunk_index, current_chunk, "recursive"))

                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n" + paragraph + "\n"
                chunk_index += 1

        if current_chunk.strip():
            chunks.append(_chunk_metadata(page, chunk_index, current_chunk, "recursive"))

    return chunks


def document_aware_chunking(pages, chunk_size=1200, overlap=150):
    chunks = []

    for page in pages:
        lines = page["text"].split("\n")
        current_section = ""
        chunk_index = 1

        for line in lines:
            line = line.strip()

            if not line:
                continue

            looks_like_heading = (
                len(line) < 90
                and not line.endswith(".")
                and any(word in line.lower() for word in ["chapter", "section", "introduction", "summary"])
            )

            if looks_like_heading and current_section.strip():
                chunks.extend(
                    fixed_chunking(
                        [{
                            "source": page.get("source", "uploaded_document.pdf"),
                            "page": page["page"],
                            "text": current_section
                        }],
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                )
                current_section = line + "\n"
                chunk_index += 1
            else:
                current_section += line + "\n"

        if current_section.strip():
            section_chunks = fixed_chunking(
                [{
                    "source": page.get("source", "uploaded_document.pdf"),
                    "page": page["page"],
                    "text": current_section
                }],
                chunk_size=chunk_size,
                overlap=overlap
            )

            for chunk in section_chunks:
                chunk["method"] = "document_aware"

            chunks.extend(section_chunks)

    return chunks


def semantic_chunking(pages, embedding_model, max_chunk_size=1200, similarity_threshold=0.62):
    chunks = []

    for page in pages:
        sentences = re.split(r"(?<=[.!?])\s+", page["text"])
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        if not sentences:
            continue

        sentence_embeddings = embedding_model.embed_texts(sentences)

        current_sentences = [sentences[0]]
        chunk_index = 1

        for index in range(1, len(sentences)):
            previous_embedding = sentence_embeddings[index - 1]
            current_embedding = sentence_embeddings[index]
            similarity = float(previous_embedding @ current_embedding)

            current_text = " ".join(current_sentences)
            next_sentence = sentences[index]

            should_continue = (
                similarity >= similarity_threshold
                and len(current_text) + len(next_sentence) <= max_chunk_size
            )

            if should_continue:
                current_sentences.append(next_sentence)
            else:
                chunk_text = " ".join(current_sentences)
                chunks.append(_chunk_metadata(page, chunk_index, chunk_text, "semantic"))

                current_sentences = [next_sentence]
                chunk_index += 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(_chunk_metadata(page, chunk_index, chunk_text, "semantic"))

    return chunks
