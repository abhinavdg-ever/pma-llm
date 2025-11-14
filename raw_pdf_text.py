import argparse
import re
from pathlib import Path

from pypdf import PdfReader

PAGE_MARKER_PATTERN = re.compile(r"^---\s*Page\s+\d+\s*---$", re.IGNORECASE)
SEE_NOTE_PATTERN = re.compile(r"\(\s*See[^)]*\)", re.IGNORECASE)
HYPHEN_BREAK_PATTERN = re.compile(r"([A-Za-z])\-\s*\n\s*([A-Za-z])")
SECTION1_START_PATTERN = re.compile(r"SECTION\s+1\s*\n\s*[A-Z]", re.IGNORECASE)
IN_WITNESS_PATTERN = re.compile(r"IN\s+WITNESS\s+WHEREOF", re.IGNORECASE)
MULTI_NEWLINE_PATTERN = re.compile(r"\n{2,}")


def extract_raw_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"--- Page {page_num} ---\n{text.strip()}")
    return "\n".join(pages)


def trim_before_section_one(text: str) -> str:
    match = SECTION1_START_PATTERN.search(text)
    if match:
        return text[match.start():]
    return text


def trim_after_in_witness(text: str) -> str:
    match = IN_WITNESS_PATTERN.search(text)
    if match:
        return text[:match.start()].rstrip() + "\n" + text[match.start():].split("\n", 1)[0].strip()
    return text


def clean_text(raw_text: str) -> str:
    lines = raw_text.splitlines()
    cleaned_lines = []
    skip_until_lowercase = False

    for line in lines:
        stripped = line.strip()
        if PAGE_MARKER_PATTERN.match(stripped):
            skip_until_lowercase = True
            continue

        if skip_until_lowercase:
            if any(ch.islower() for ch in stripped):
                skip_until_lowercase = False
            else:
                continue

        if stripped:
            cleaned_lines.append(stripped)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = trim_before_section_one(cleaned_text)
    cleaned_text = trim_after_in_witness(cleaned_text)
    cleaned_text = HYPHEN_BREAK_PATTERN.sub(r"\1\2", cleaned_text)
    cleaned_text = SEE_NOTE_PATTERN.sub("", cleaned_text)
    cleaned_text = MULTI_NEWLINE_PATTERN.sub("\n", cleaned_text)
    return cleaned_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract raw and cleaned text from a PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--raw-out", type=Path, required=True, help="Output file for raw text")
    parser.add_argument("--clean-out", type=Path, required=True, help="Output file for cleaned text")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"File not found: {args.pdf}")

    raw_text = extract_raw_text(args.pdf)
    cleaned_text = clean_text(raw_text)

    args.raw_out.write_text(raw_text, encoding="utf-8")
    args.clean_out.write_text(cleaned_text, encoding="utf-8")

    print(f"Saved raw text to {args.raw_out.resolve()}")
    print(f"Saved cleaned text to {args.clean_out.resolve()}")


if __name__ == "__main__":
    main()
