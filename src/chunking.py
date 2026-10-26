from pathlib import Path
from transformers import AutoTokenizer
from src import config
from src.utils import load_jsonl, save_jsonl

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    return _tokenizer

def split_into_chunks(text, chunk_size, overlap):
    """Pack whole lines (table rows / paragraphs) into chunks of <= chunk_size tokens.
    A single line longer than chunk_size (a long paragraph) is cut by tokens instead."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return []
    enc = get_tokenizer()(lines, add_special_tokens=False,
                          return_offsets_mapping=True, verbose=False)
    pieces = []                                   # (text, n_tokens)
    for line, ids, offs in zip(lines, enc["input_ids"], enc["offset_mapping"]):
        n = len(ids)
        if n <= chunk_size:
            pieces.append((line, n))
            continue
        for s in range(0, n, chunk_size - overlap):
            e = min(s + chunk_size, n)
            pieces.append((line[offs[s][0]:offs[e - 1][1]], e - s))
            if e == n:
                break

    chunks, current, size = [], [], 0
    for line, n in pieces:
        if current and size + n > chunk_size:
            chunks.append("\n".join(t for t, _ in current))
            last = current[-1]
            current = [last] if last[1] <= overlap else []   # repeat last short line for continuity
            size = sum(k for _, k in current)
        current.append((line, n))
        size += n
    if current:
        chunks.append("\n".join(t for t, _ in current))
    return chunks

def build_chunks():
    pages = load_jsonl(config.PAGES_PATH)
    chunks = []
    for page in pages:
        pieces = split_into_chunks(
            page["text"], config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
        )
        rows = [l.strip() for l in page["text"].split("\n") if l.strip().startswith("|")]
        table_head = "\n".join(rows[:2])          # the table's column-header rows
        header = f"{page['report']} | PDF page {page['page']}"
        if page.get("title"):
            header += f" | {page['title']}"
        stem = Path(page["source_file"]).stem
        for i, piece in enumerate(pieces):
            if i > 0 and table_head and piece.startswith("|"):
                piece = table_head + "\n" + piece   # later table chunks keep their column headers
            chunks.append({
                "chunk_id": f"{stem}_p{page['page']:03d}_c{i:02d}",
                "report": page["report"],
                "source_file": page["source_file"],
                "page": page["page"],
                "printed_page": page.get("printed_page"),
                "text": header + "\n" + piece,
            })
    save_jsonl(chunks, config.CHUNKS_PATH)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    return chunks
