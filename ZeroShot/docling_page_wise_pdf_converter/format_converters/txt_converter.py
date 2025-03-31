# docling-page-wise-pdf-converter/format_converters/txt_converter.py
from pathlib import Path
from typing import Dict
from docling_core.types.doc import TableItem, PictureItem, TextItem
from .base_converter import BaseConverter

class TxtConverter(BaseConverter):
    def convert_to_format(self, doc, pdf_path: Path, output_dir: Path) -> Dict[int, str]:
        """
        Converts the document to TXT format.
        """
        pages_content = {}
        for page_number in doc.pages.keys():
            text_sections = []
            text_sections.append(f"\n{'='*3}Page {page_number}{'='*3}\n")


            for item in doc.iterate_items(page_no=page_number):
                if isinstance(item[0], TableItem):
                    text_sections.append(item[0].export_to_dataframe().to_string())
                elif isinstance(item[0], PictureItem): # todo: recognize image into text using LLM
                    caption = item[0].caption_text(doc)
                    text_sections.append(f"[Image: {caption}]" if caption else "[Image]")
                elif isinstance(item[0], TextItem):
                    text_sections.append(item[0].text)

            pages_content[page_number] = "\n".join(text_sections)
        return pages_content

    def save_with_original_extension(self, page_contents: Dict[int, str], pdf_path: Path, output_dir: Path, doc): # Added 'doc' here (but not used)
        """
        Saves the converted content to text files (.txt).
        """
        output_path = output_dir / f"{pdf_path.stem}.txt"
        text_sections_all = []
        for page_number, content in page_contents.items():
            text_sections_all.append(f"\n{'='*80}\nPage {page_number}\n{'='*80}\n")
            text_sections_all.append(content)

        final_text = "\n".join(text_sections_all)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)