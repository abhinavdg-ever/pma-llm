import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader

PAGE_MARKER_PATTERN = re.compile(r"^---\s*Page\s+\d+\s*---$", re.IGNORECASE)
SEE_NOTE_PATTERN = re.compile(r"\(\s*See[^)]*\)", re.IGNORECASE)
HYPHEN_BREAK_PATTERN = re.compile(r"([A-Za-z])\-\s*\n\s*([A-Za-z])")
SECTION1_START_PATTERN = re.compile(r"SECTION\s+1\s*\n\s*[A-Z]", re.IGNORECASE)
IN_WITNESS_PATTERN = re.compile(r"IN\s+WITNESS\s+WHEREOF", re.IGNORECASE)
MULTI_NEWLINE_PATTERN = re.compile(r"\n{2,}")
SECTION_HEADER_RE = re.compile(r"^Section\s+(\d+)\b", re.IGNORECASE)
SECTION_UPPER_LINE_RE = re.compile(r"^SECTION\s+(\d+)\b")
CLAUSE_RE = re.compile(r"^(?P<id>\d+(?:\.\d+)+)(?P<trailer>[A-Za-z]*)\s*(?P<text>.*)")


def clean_field(value: str) -> str:
    cleaned = value.replace('""', '"').strip().strip('"')
    return cleaned


def is_uppercase_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if not any(ch.isalpha() for ch in stripped):
        return False
    return stripped.upper() == stripped


def normalize_title_line(line: str, section_id: str) -> str:
    stripped = line.strip()
    pattern = rf"^SECTION\s+{section_id}\b"
    return re.sub(pattern, "", stripped, flags=re.IGNORECASE).strip()


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


def extract_section_from_clause_id(clause_id: str) -> Optional[str]:
    """Extract section number from clause ID (e.g., '1.8' -> '1', '12.1' -> '12')."""
    parts = clause_id.split(".", 1)
    if parts and parts[0].isdigit():
        return parts[0]
    return None


def parse_contract(text: str, raw_text: Optional[str] = None) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    lines = text.splitlines()
    sections: List[Dict[str, str]] = []
    clauses: List[Dict[str, str]] = []

    sections_seen: Dict[str, str] = {}
    clauses_seen = set()

    current_section: Optional[str] = None
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        section_match = SECTION_HEADER_RE.match(stripped)
        upper_section_match = SECTION_UPPER_LINE_RE.match(stripped)
        if upper_section_match:
            section_id = upper_section_match.group(1)
            current_section = section_id
            if section_id not in sections_seen:
                sections.append(
                    {
                        "section_id": section_id,
                        "section_title": "",
                        "start_page": "",
                        "end_page": "",
                    }
                )
                sections_seen[section_id] = ""
            i += 1
            continue

        if section_match:
            section_id = section_match.group(1)
            i += 1

            title_lines: List[str] = []
            while i < len(lines):
                candidate = lines[i].strip()
                if not candidate:
                    i += 1
                    continue
                if SECTION_HEADER_RE.match(candidate):
                    break
                if CLAUSE_RE.match(candidate):
                    break
                if is_uppercase_line(candidate):
                    normalized = normalize_title_line(candidate, section_id)
                    if normalized:
                        title_lines.append(normalized)
                    i += 1
                    continue
                break

            section_title = clean_field(" ".join(title_lines))
            if section_id not in sections_seen:
                sections.append(
                    {
                        "section_id": section_id,
                        "section_title": section_title,
                        "start_page": "",
                        "end_page": "",
                    }
                )
                sections_seen[section_id] = section_title
            else:
                if not sections_seen[section_id]:
                    sections_seen[section_id] = section_title
                    for section in sections:
                        if section["section_id"] == section_id:
                            section["section_title"] = section_title
                            break

            current_section = section_id
            continue

        clause_match = CLAUSE_RE.match(stripped)
        if clause_match:
            clause_id = clause_match.group("id") + clause_match.group("trailer")
            if clause_id in clauses_seen:
                i += 1
                continue

            # Extract section from clause_id itself (e.g., "1.8" -> section "1")
            clause_section = extract_section_from_clause_id(clause_id)
            # Use clause_section if available, otherwise fall back to current_section
            assigned_section = clause_section if clause_section else current_section

            if not assigned_section:
                i += 1
                continue

            text_parts: List[str] = []
            initial_text = clause_match.group("text").strip()
            if initial_text:
                text_parts.append(initial_text)

            i += 1
            while i < len(lines):
                lookahead = lines[i].strip()
                if not lookahead:
                    i += 1
                    continue
                if SECTION_HEADER_RE.match(lookahead):
                    break
                if CLAUSE_RE.match(lookahead):
                    break
                text_parts.append(lookahead)
                i += 1

            clause_text = clean_field(" ".join(text_parts))
            clauses.append(
                {
                    "section_id": assigned_section,
                    "clause_id": clause_id,
                    "text": clause_text,
                }
            )
            clauses_seen.add(clause_id)
            continue

        i += 1

    # Add page numbers to sections by finding them in raw text
    if raw_text:
        raw_lines = raw_text.splitlines()
        current_page = None
        section_page_map: Dict[str, int] = {}
        
        for line in raw_lines:
            # Track current page
            page_match = PAGE_MARKER_PATTERN.match(line.strip())
            if page_match:
                page_num_match = re.search(r"Page\s+(\d+)", line, re.IGNORECASE)
                if page_num_match:
                    current_page = int(page_num_match.group(1))
                continue
            
            # Check for section headers (both "Section X" and "SECTION X")
            if current_page is not None:
                section_match = SECTION_HEADER_RE.match(line.strip())
                upper_section_match = SECTION_UPPER_LINE_RE.match(line.strip())
                section_id = None
                if section_match:
                    section_id = section_match.group(1)
                elif upper_section_match:
                    section_id = upper_section_match.group(1)
                
                if section_id and section_id not in section_page_map:
                    section_page_map[section_id] = current_page
        
        # Update sections with page numbers
        for section in sections:
            section_id = section["section_id"]
            if section_id in section_page_map:
                section["start_page"] = str(section_page_map[section_id])
        
        # Calculate end_page (same as next section's start_page, or empty if last)
        sorted_sections = sorted(sections, key=lambda s: (int(s["section_id"]) if s["section_id"].isdigit() else 999))
        for idx, section in enumerate(sorted_sections):
            if idx + 1 < len(sorted_sections):
                next_section = sorted_sections[idx + 1]
                if next_section["start_page"]:
                    section["end_page"] = next_section["start_page"]
            # If no end_page set and it's the last section, leave it empty

    return sections, clauses


def merge_sections_clauses(
    sections: List[Dict[str, str]], clauses: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    title_by_section = {section["section_id"]: section["section_title"] for section in sections}
    merged: List[Dict[str, str]] = []
    for clause in clauses:
        merged.append(
            {
                "section_id": clause["section_id"],
                "section_title": title_by_section.get(clause["section_id"], ""),
                "clause_id": clause["clause_id"],
                "text": clause["text"],
            }
        )
    return merged


def get_main_clause_id(clause_id: str) -> str:
    """Extract main clause ID: section number + first digit after the dot.
    
    Works for all sections (single or multi-digit section numbers).
    
    Examples:
    - '1.1' -> '1.1'
    - '1.11' -> '1.1' (section 1, first digit after dot is 1)
    - '1.12' -> '1.1' (section 1, first digit after dot is 1)
    - '1.131' -> '1.1' (section 1, first digit after dot is 1)
    - '12.1' -> '12.1' (section 12, first digit after dot is 1)
    - '12.11' -> '12.1' (section 12, first digit after dot is 1)
    - '12.12' -> '12.1' (section 12, first digit after dot is 1)
    - '12.2' -> '12.2' (section 12, first digit after dot is 2)
    """
    parts = clause_id.split(".", 1)  # Split only on first dot
    if len(parts) >= 2:
        section_num = parts[0]  # Can be "1", "12", "20", etc.
        after_dot = parts[1]
        # Extract first digit/character after the dot
        first_digit = after_dot[0] if after_dot else ""
        return f"{section_num}.{first_digit}"
    return clause_id


def combine_clauses_by_main(
    sections: List[Dict[str, str]], clauses: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Group clauses by main clause ID and combine their texts."""
    title_by_section = {section["section_id"]: section["section_title"] for section in sections}
    
    # Group clauses by section_id and main_clause_id
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for clause in clauses:
        section_id = clause["section_id"]
        clause_id = clause["clause_id"]
        main_clause_id = get_main_clause_id(clause_id)
        key = (section_id, main_clause_id)
        
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(clause)
    
    # Build combined rows
    combined: List[Dict[str, str]] = []
    for (section_id, main_clause_id), clause_list in sorted(grouped.items()):
        # Sort clauses by their full ID to maintain order
        # Handle numeric parts and letter suffixes (e.g., "1.1", "1.11", "1.1a")
        def sort_key(c: Dict[str, str]) -> Tuple:
            parts = c["clause_id"].split(".")
            result = []
            for part in parts:
                # Extract numeric prefix and optional letter suffix
                match = re.match(r"(\d+)([a-zA-Z]*)", part)
                if match:
                    num = int(match.group(1))
                    suffix = match.group(2) or ""
                    result.append((num, suffix))
                else:
                    result.append((0, part))
            return tuple(result)
        
        clause_list.sort(key=sort_key)
        
        sub_clause_ids = [c["clause_id"] for c in clause_list]
        texts = [c["text"] for c in clause_list]
        combined_text = "\n".join(texts)
        
        combined.append(
            {
                "section_id": section_id,
                "section_title": title_by_section.get(section_id, ""),
                "main_clause_id": main_clause_id,
                "sub_clause_ids": ",".join(sub_clause_ids),
                "text": clean_field(combined_text),
            }
        )
    
    # Sort combined rows by section_id as int
    combined.sort(key=lambda x: (int(x["section_id"]) if x["section_id"].isdigit() else 999, x["main_clause_id"]))
    
    return combined


def write_csv(output_path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, clean, and parse a contract PDF into section, clause, and merged datasets."
    )
    parser.add_argument("pdf", type=Path, help="Path to the contract PDF file.")
    parser.add_argument(
        "--raw-out",
        type=Path,
        help="Optional path to write the raw extracted text.",
    )
    parser.add_argument(
        "--clean-out",
        type=Path,
        help="Optional path to write the cleaned text.",
    )
    parser.add_argument(
        "--sections-out",
        type=Path,
        required=True,
        help="Output CSV file for section metadata.",
    )
    parser.add_argument(
        "--clauses-out",
        type=Path,
        required=True,
        help="Output CSV file for clause text.",
    )
    parser.add_argument(
        "--merged-out",
        type=Path,
        help="Optional output CSV file for merged section/clauses data.",
    )
    parser.add_argument(
        "--combined-out",
        type=Path,
        help="Optional output CSV file for clauses combined by main clause ID.",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"File not found: {args.pdf}")

    raw_text = extract_raw_text(args.pdf)
    if args.raw_out:
        args.raw_out.write_text(raw_text, encoding="utf-8")

    cleaned_text = clean_text(raw_text)
    if args.clean_out:
        args.clean_out.write_text(cleaned_text, encoding="utf-8")

    sections, clauses = parse_contract(cleaned_text, raw_text=raw_text)

    if not sections:
        raise SystemExit("No sections were detected in the provided text.")

    write_csv(args.sections_out, ["section_id", "section_title", "start_page", "end_page"], sections)
    write_csv(args.clauses_out, ["section_id", "clause_id", "text"], clauses)

    if args.merged_out:
        merged_rows = merge_sections_clauses(sections, clauses)
        write_csv(args.merged_out, ["section_id", "section_title", "clause_id", "text"], merged_rows)

    combined_rows = None
    if args.combined_out:
        combined_rows = combine_clauses_by_main(sections, clauses)
        write_csv(
            args.combined_out,
            ["section_id", "section_title", "main_clause_id", "sub_clause_ids", "text"],
            combined_rows,
        )

    print(f"Wrote {len(sections)} sections to {args.sections_out.resolve()}")
    print(f"Wrote {len(clauses)} clauses to {args.clauses_out.resolve()}")
    if args.merged_out:
        print(f"Wrote {len(clauses)} merged rows to {args.merged_out.resolve()}")
    if args.combined_out and combined_rows:
        print(f"Wrote {len(combined_rows)} combined clause groups to {args.combined_out.resolve()}")


if __name__ == "__main__":
    main()

