from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import pdfplumber
import io
import csv
import pandas as pd
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK_SIZE = 1000  # characters for text chunk


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename

    try:
        if filename.lower().endswith(".pdf"):
            chunks = extract_pdf_chunks(content, filename)
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            chunks = extract_image_chunks(content, filename)
        elif filename.lower().endswith(".csv"):
            chunks = extract_csv_chunks(content, filename)
        elif filename.lower().endswith((".xls", ".xlsx")):
            chunks = extract_excel_chunks(content, filename)
        else:
            return JSONResponse(
                status_code=400, content={"error": "Unsupported file type"}
            )

        return {"filename": filename, "chunks": chunks}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def chunk_text(text, max_size=CHUNK_SIZE):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        if end > len(text):
            end = len(text)
        else:
            # Try to cut on last space before max_size to avoid cutting words
            last_space = text.rfind(" ", start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def extract_pdf_chunks(pdf_bytes, filename):
    chunks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        chunk_index = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                # OCR fallback
                image = page.to_image(resolution=300).original
                page_text = pytesseract.image_to_string(image)

            text_chunks = chunk_text(page_text, CHUNK_SIZE)
            for chunk in text_chunks:
                chunks.append(
                    {
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "page_number": page_num,
                        "type": "text",
                        "content": chunk,
                    }
                )
                chunk_index += 1

            # Extract tables
            for table_index, table in enumerate(page.extract_tables()):
                # Clean table cells
                cleaned_table = [
                    [cell.strip() if cell else "" for cell in row] for row in table
                ]
                chunks.append(
                    {
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "page_number": page_num,
                        "type": "table",
                        "content": cleaned_table,
                        "table_index": table_index,
                    }
                )
                chunk_index += 1
    return chunks


def extract_image_chunks(image_bytes, filename):
    text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
    chunks = []
    for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
        chunks.append(
            {
                "filename": filename,
                "chunk_index": i,
                "page_number": None,
                "type": "text",
                "content": chunk,
            }
        )
    return chunks


def extract_csv_chunks(csv_bytes, filename):
    decoded = csv_bytes.decode("utf-8").splitlines()
    reader = csv.reader(decoded)
    table = [row for row in reader]

    # Convert entire CSV to text
    text = "\n".join([", ".join(row) for row in table])

    chunks = []
    # Text chunk(s)
    for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
        chunks.append(
            {
                "filename": filename,
                "chunk_index": i,
                "page_number": None,
                "type": "text",
                "content": chunk,
            }
        )

    # Table chunk (whole CSV as one table)
    chunks.append(
        {
            "filename": filename,
            "chunk_index": len(chunks),
            "page_number": None,
            "type": "table",
            "content": table,
            "table_index": 0,
        }
    )
    return chunks


def extract_excel_chunks(excel_bytes, filename):
    xls = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
    chunks = []
    chunk_index = 0

    for sheet_name, df in xls.items():
        # Text chunk from CSV-formatted sheet
        text = df.to_csv(index=False)
        for chunk in chunk_text(text, CHUNK_SIZE):
            chunks.append(
                {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "page_number": None,
                    "type": "text",
                    "content": chunk,
                    "sheet_name": sheet_name,
                }
            )
            chunk_index += 1

        # Table chunk (whole sheet as one table)
        table = df.fillna("").astype(str).values.tolist()
        chunks.append(
            {
                "filename": filename,
                "chunk_index": chunk_index,
                "page_number": None,
                "type": "table",
                "content": table,
                "sheet_name": sheet_name,
                "table_index": 0,
            }
        )
        chunk_index += 1

    return chunks
