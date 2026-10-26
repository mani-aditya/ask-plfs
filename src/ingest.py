import re
import urllib.request

import pymupdf

from src import config
from src.utils import save_jsonl
# matches "Annual Report, PLFS, 2025 xx" and "Annual Report, PLFS, 2025 10" (printed page number)
PRINTED_RE = re.compile(r"Annual Report, PLFS, 2025\s+([ivxlc]+|\d+)\b")
# first line that looks like "Table 4: ..." is used as the page title
TITLE_RE = re.compile(r"^[#*\s]*((?:Table|Statement|Annexure)\b[^\n]{0,150})", re.M)
HEADING_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$", re.M)   # fallback title: first markdown heading

def clean_text(text):
    text = text.replace("\x00", " ").replace("<br>", " ")
    text = re.sub(r"</?mark>", "", text)                       # parser highlight tags
    text = text.replace("**", "")                              # bold markers
    text = re.sub(r"^\|[-:| ]+\|\s*$", "", text, flags=re.M)   # drop |---|---| rows (must come BEFORE the dash line)
    text = re.sub(r"-{3,}", " ", text)                         # long dash runs
    text = re.sub(r"(\|[ \t]*){2,}", "| ", text)               # empty cells; [ \t] not \s, so rows stay on separate lines
    text = re.sub(r"-\n(?=[a-z])", "", text)                   # re-join hyphenated words
    text = re.sub(r"^_Summar[^\n]*$", "", text, flags=re.M)    # garbled running header
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_page_texts(pdf_path):
    """One text string per PDF page."""
    if config.USE_MARKDOWN:
        import pymupdf4llm
        parts = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        return [part["text"] for part in parts]
    doc = pymupdf.open(pdf_path)
    return [page.get_text("text", sort=True) for page in doc]

def extract_pages(pdf_path, report_label):
    pages, skipped = [], []
    for page_index, raw in enumerate(get_page_texts(pdf_path)):
        text = clean_text(raw)
        match = PRINTED_RE.search(text)
        printed_page = match.group(1) if match else None
        text = PRINTED_RE.sub("", text).strip()          # remove the repeated header/footer
        title = TITLE_RE.search(text) or HEADING_RE.search(text)
        if len(text) < config.MIN_PAGE_CHARS:
            skipped.append((page_index + 1, "too short")); continue
        if english_ratio(text) < config.MIN_ENGLISH_RATIO:
            skipped.append((page_index + 1, "not English")); continue
        pages.append({
            "report": report_label,
            "source_file": pdf_path.name,
            "page": page_index + 1,                      # PDF page
            "printed_page": printed_page,                # page number printed on the page
            "title": title.group(1).strip(" *#") if title else None,
            "text": text,
        })
    print(f"{pdf_path.name}: kept {len(pages)}, skipped {len(skipped)}")
    print("First skipped pages:", skipped[:20])
    return pages

def download_pdf(url, dest):
    """Download a PDF, pretending to be a normal browser."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())





def english_ratio(text):
    """Share of letters that are plain ASCII (Hindi text scores near 0)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)





def build_pages():
    """Download (if missing) and parse every report listed in config.REPORTS."""
    all_pages = []
    for file_name, meta in config.REPORTS.items():
        pdf_path = config.DATA_RAW / file_name
        if not pdf_path.exists():
            print(f"Downloading {file_name} ...")
            download_pdf(meta["url"], pdf_path)
        pages = extract_pages(pdf_path, meta["label"])

        all_pages.extend(pages)
    save_jsonl(all_pages, config.PAGES_PATH)
    return all_pages
