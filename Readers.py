import fitz  # PyMuPDF
from docx import Document
from docx.oxml.ns import qn
from pptx import Presentation

# OCR is optional. If pytesseract/Pillow/tesseract aren't installed on this
# machine, we fail soft and just keep behaving like before (no OCR fallback)
# instead of crashing the whole extraction pipeline.
try:
    import pytesseract
    from PIL import Image, ImageOps
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Below this many non-whitespace characters, we treat the PDF's text layer
# as "empty" (e.g. scanned/image-only PDF) and try OCR instead.
MIN_CHARS_BEFORE_OCR = 30

# 300 dpi + grayscale/autocontrast made a real difference on colorful,
# graphic-heavy PDFs (marketing flyers, certificates with logo grids) where
# 200 dpi + raw color was silently dropping entire columns of text --
# not just noisier output, but actually missing content.
OCR_DPI = 300


def _ocr_pdf(path):
    """Rasterize each page and OCR it. Returns '' on any failure."""

    if not OCR_AVAILABLE:
        return ""

    text = ""

    try:
        pdf = fitz.open(path)

        for page in pdf:
            pix = page.get_pixmap(dpi=OCR_DPI)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

            # Grayscale + autocontrast before OCR. Tesseract's own
            # binarization can fail on saturated/colorful backgrounds
            # (e.g. bright green marketing graphics), silently dropping
            # whole blocks of text rather than just adding noise.
            image = ImageOps.grayscale(image)
            image = ImageOps.autocontrast(image)

            text += pytesseract.image_to_string(image)
            text += "\n"

        pdf.close()

    except Exception as e:
        print(f"Error OCR-reading PDF {path}: {e}")
        return ""

    return text


def read_pdf(path):
    text = ""

    try:
        pdf = fitz.open(path)

        for page in pdf:
            text += page.get_text()
            text += "\n"

        pdf.close()

    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return text

    # Text layer looks empty/near-empty (likely a scanned or image-only
    # PDF) -> fall back to OCR. If OCR isn't available or also fails,
    # we just keep whatever (possibly empty) text we already have.
    if len(text.strip()) < MIN_CHARS_BEFORE_OCR:

        ocr_text = _ocr_pdf(path)

        if len(ocr_text.strip()) > len(text.strip()):
            text = ocr_text

    return text


def read_docx(path):
    text = ""

    try:
        document = Document(path)

        for paragraph in document.paragraphs:
            text += paragraph.text
            text += "\n"

        # Paragraphs alone miss anything laid out in a table (invoices,
        # spec sheets, resumes with tabular sections, etc).
        for table in document.tables:
            for row in table.rows:
                cells_text = [cell.text.strip() for cell in row.cells]
                text += " | ".join(cells_text)
                text += "\n"

        # Paragraphs AND tables both miss text boxes (floating shapes) --
        # a very common way invoice/flyer-style templates lay out content.
        # Text box content lives in <w:txbxContent> elements that aren't
        # reachable via document.paragraphs or document.tables at all.
        for txbx in document.element.body.iter(qn("w:txbxContent")):
            for t_node in txbx.iter(qn("w:t")):
                if t_node.text:
                    text += t_node.text
            text += "\n"

    except Exception as e:
        print(f"Error reading DOCX {path}: {e}")

    return text


def read_pptx(path):
    text = ""

    try:
        presentation = Presentation(path)

        for slide in presentation.slides:
            for shape in slide.shapes:

                if hasattr(shape, "text"):
                    text += shape.text
                    text += "\n"

                # Tables in pptx shapes aren't picked up by shape.text.
                if shape.has_table:
                    for row in shape.table.rows:
                        cells_text = [cell.text.strip() for cell in row.cells]
                        text += " | ".join(cells_text)
                        text += "\n"

    except Exception as e:
        print(f"Error reading PPTX {path}: {e}")

    return text


def read_txt(path):
    text = ""

    try:
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading TXT {path}: {e}")

    except Exception as e:
        print(f"Error reading TXT {path}: {e}")

    return text