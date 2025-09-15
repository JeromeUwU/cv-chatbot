
import os, io, re, json
from pypdf import PdfReader
import streamlit as st
def extract_text_from_pdf_bytes(b: bytes) -> str:
    """Extraction texte basique depuis un PDF (fallback si pas d'Europass JSON)."""
    try:
        reader = PdfReader(io.BytesIO(b))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n\n".join(pages).strip()
    except Exception:
        return ""

def try_extract_europass_json(b: bytes):
    """
    Certains PDF Europass contiennent un JSON encodé en UTF-16 dans /ecv-data <FEFF...>.
    On tente de l’extraire proprement.
    """
    try:
        m = re.search(br"/ecv-data\s*<([0-9A-Fa-f]+)>", b)
        if not m:
            return None
        hex_payload = m.group(1)
        raw = bytes.fromhex(hex_payload.decode("ascii"))
        # FEFF = BOM UTF-16; on tente 'utf-16-be' (souvent le cas)
        text = raw.decode("utf-16-be", errors="ignore").strip()
        return json.loads(text)
    except Exception:
        return None

def europass_json_to_markdown(data: dict) -> str:
    """
    Version générique: on produit un markdown minimal + JSON pliable.
    (Si tu veux, on fera une mise en forme sectionnée plus tard.)
    """
    # On essaie naïvement de récupérer un nom si présent
    name = None
    for key in ("fullName", "name", "personName"):
        if isinstance(data.get(key), str):
            name = data[key]
            break
        if isinstance(data.get(key), dict):
            # ex: {"firstName": "...", "lastName": "..."}
            fn = data[key].get("firstName") or data[key].get("givenName")
            ln = data[key].get("lastName") or data[key].get("familyName")
            if fn or ln:
                name = " ".join([p for p in [fn, ln] if p])
                break

    header = f"# CV (Europass import)\n\n**Nom** : {name}\n" if name else "# CV (Europass import)\n"
    raw_json = json.dumps(data, ensure_ascii=False, indent=2)
    return (
        header
        + "\n<details><summary>Données brutes (JSON)</summary>\n\n"
        + "```json\n"
        + raw_json
        + "\n```\n\n</details>\n"
    )

def load_cv_text() -> str:
    """
    1) L’utilisateur peut uploader un .md/.txt/.pdf
    2) Sinon on cherche des fichiers locaux (cv.md, cv.txt, cv.pdf)
    3) Fallback: texte vide
    """
    uploaded = st.sidebar.file_uploader("📄 Importer mon CV (.md, .txt, .pdf)", type=["md", "txt", "pdf"])
    if uploaded:
        suffix = uploaded.name.split(".")[-1].lower()
        content = uploaded.read()
        if suffix in ("md", "txt"):
            return content.decode("utf-8", errors="ignore")
        if suffix == "pdf":
            # 1) Essai extraction “texte simple”
            txt = extract_text_from_pdf_bytes(content)
            if txt:
                return txt
            # 2) Essai extraction JSON Europass
            data = try_extract_europass_json(content)
            if data:
                return europass_json_to_markdown(data)
            return ""  # rien trouvé

    # Fichiers locaux (utile en dev)
    for path in ("cv.md", "cv.txt", "cv.pdf"):
        if os.path.exists(path):
            if path.endswith(".pdf"):
                with open(path, "rb") as f:
                    b = f.read()
                txt = extract_text_from_pdf_bytes(b)
                if txt:
                    return txt
                data = try_extract_europass_json(b)
                if data:
                    return europass_json_to_markdown(data)
            else:
                return open(path, "r", encoding="utf-8", errors="ignore").read()

    return ""
