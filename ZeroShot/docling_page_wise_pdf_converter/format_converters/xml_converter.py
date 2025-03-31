# docling-page-wise-pdf-converter/format_converters/xml_converter.py
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
from docling_core.types.doc import TableItem, PictureItem, TextItem
from .base_converter import BaseConverter

class XmlConverter(BaseConverter):
    def convert_to_format(self, doc, pdf_path: Path, output_dir: Path) -> Dict[int, str]:
        """
        Converts the document to XML format.
        """
        pages_content = {}
        for page_number in doc.pages.keys():
            pages_content[page_number] = self._convert_page_to_xml(doc, page_number)
        return pages_content

    def _convert_page_to_xml(self, doc, page_number: int) -> str:
        """Helper function to convert a single page to XML."""
        page_element = ET.Element("page")
        page_element.set("number", str(page_number))

        for item, _ in doc.iterate_items(page_no=page_number):
            if isinstance(item, TextItem):
                item_element = ET.SubElement(page_element, "text")
                item_element.text = item.text
            elif isinstance(item, TableItem):
                item_element = ET.SubElement(page_element, "table")
                try:
                    item_element.text = item.export_to_dataframe().to_string()
                except:
                    item_element.text = "[Table content not extractable]"
            elif isinstance(item, PictureItem):
                item_element = ET.SubElement(page_element, "image")
                caption = item.caption_text(doc) if hasattr(item, 'caption_text') else None
                if caption:
                    item_element.set("caption", caption)

        return ET.tostring(page_element, encoding="unicode")

    def save_with_original_extension(self, page_contents: Dict[int, str], pdf_path: Path, output_dir: Path, doc): # Added 'doc' here (but not used)
        """
        Saves the converted content to xml files (.xml).
        """
        output_path = output_dir / f"{pdf_path.stem}.xml"

        root = ET.Element("document")
        root.set("name", pdf_path.stem)

        for page_number, content in page_contents.items():
            # Content is already XML, so parse it and append
            page_element = ET.fromstring(content)
            root.append(page_element)

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)