import fitz  # PyMuPDF

PDF_PATH = r"C:\Users\megha\Downloads\RAG\All New Amaze - 3rd Gen (2024~).pdf"

CHAPTERS = [
    {"title": "Before Driving", "page_start": 13, "page_end": 89},
    {"title": "Driving Operation", "page_start": 90, "page_end": 129},
    {"title": "Controls", "page_start": 130, "page_end": 147},
    {"title": "Safety Driving Assist System", "page_start": 148, "page_end": 187},
    {"title": "About Your Instrument Panel", "page_start": 188, "page_end": 209},
    {"title": "Maintenance", "page_start": 210, "page_end": 239},
    {"title": "Handling the Unexpected", "page_start": 240, "page_end": 274},
    {"title": "Vehicle Information", "page_start": 275, "page_end": 290},
]


def extract_chapter_text(pdf_path=PDF_PATH, chapters=CHAPTERS):
    doc = fitz.open(pdf_path)
    extracted = []

    for ch in chapters:
        start = ch["page_start"] - 1
        end = ch["page_end"] - 1

        pages = []
        for p in range(start, end + 1):
            text = doc.load_page(p).get_text("text")
            clean = " ".join(text.split())
            pages.append(clean)

        extracted.append({
            "chapter": ch["title"],
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "text": "\n".join(pages)
        })

    return extracted
