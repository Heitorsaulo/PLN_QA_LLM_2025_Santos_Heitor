from typing import List
from pathlib import Path
from core.models import Document
import docx2txt
import PyPDF2

class DocumentLoader:
    @staticmethod
    def load_docx2txt(file_path: str) -> str:
        doc = docx2txt.process(file_path)
        return doc

    @staticmethod
    def load_pdf(file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    @staticmethod
    def load_documents(paths: List[str]) -> List[Document]:
        docs = []
        for path in paths:
            ext = Path(path).suffix.lower()
            if ext == '.pdf':
                content = DocumentLoader.load_pdf(path)
            elif ext == '.docx':
                content = DocumentLoader.load_docx2txt(path)
            else:
                continue
            docs.append(Document(id=str(path), content=content))
        return docs
