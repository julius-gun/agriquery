# docling-page-wise-pdf-converter/format_converters/markdown_converter.py
from pathlib import Path
from typing import Dict
from .base_converter import BaseConverter

class MarkdownConverter(BaseConverter):
    def convert_to_format(self, doc, pdf_path: Path, output_dir: Path) -> Dict[int, str]:
        """
        Converts the document to Markdown format.
        """
        pages_content = {}
        for page_number in doc.pages.keys():
            pages_content[page_number] = doc.export_to_markdown(
                page_no=page_number
            )
        return pages_content

    def save_with_original_extension(self, page_contents: Dict[int, str], pdf_path: Path, output_dir: Path, doc): # Added 'doc' here (but not used)
        """
        Saves the converted content to markdown files (.md).
        """
        output_path = output_dir / f"{pdf_path.stem}.md"
        with open(output_path, "w", encoding='utf-8') as f:
            for page_number, content in page_contents.items():
                f.write(f"## Page {page_number}\n\n")
                f.write(f"{content}\n\n")