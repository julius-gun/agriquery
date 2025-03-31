# docling-page-wise-pdf-converter/format_converters/json_converter.py
import json
from pathlib import Path
from typing import Dict
from docling_core.types.doc import TableItem, PictureItem, TextItem
from .base_converter import BaseConverter

class JsonConverter(BaseConverter):
    def convert_to_format(self, doc, pdf_path: Path, output_dir: Path) -> Dict[int, str]:
        """
        Converts the document to JSON format.
        """
        pages_content = {}
        for page_number in doc.pages.keys():
            pages_content[page_number] = self._get_json_for_page(doc, page_number)
        return pages_content

    def _get_dict_for_page(self, doc, page_number: int) -> Dict:
        """
        Extract content of a specific page as a dictionary, handling different item types.
        """
        page_dict = {"page_number": page_number, "items": []}
        for item, parent in doc.iterate_items(page_no=page_number):
            item_dict = {"type": item.__class__.__name__} # Add type information

            if isinstance(item, TextItem):
                item_dict["text"] = item.text
            elif isinstance(item, TableItem):
                try:
                    df = item.export_to_dataframe()
                    # Make DataFrame columns unique
                    if not df.columns.is_unique:
                        import pandas as pd
                        new_columns = []
                        seen_columns = {}
                        for col in df.columns:
                            name = str(col) # Convert to string in case of non-string columns
                            if name in seen_columns:
                                seen_columns[name] += 1
                                new_col_name = f"{name}_{seen_columns[name]}"
                            else:
                                seen_columns[name] = 1
                                new_col_name = name
                            new_columns.append(new_col_name)
                        df.columns = new_columns
                    item_dict["table_data"] = df.to_dict()
                except Exception as e:
                    item_dict["table_data"] = "Table content not extractable"
            elif isinstance(item, PictureItem):
                item_dict["caption"] = item.caption_text(doc) if hasattr(item, 'caption_text') else None
                # We can add more PictureItem attributes here if needed, e.g., image path, etc.

            page_dict["items"].append(item_dict)
        return page_dict
    
    def _get_json_for_page(self, doc, page_number: int) -> str:
        """
        Extract content of a specific page as a JSON string.
        """
        return json.dumps(self._get_dict_for_page(doc, page_number), indent=2)

    def save_with_original_extension(self, page_contents: Dict[int, str], pdf_path: Path, output_dir: Path, doc): # Added 'doc' here (but not used)
        """
        Saves the converted content to json files (.json).
        """
        output_path = output_dir / f"{pdf_path.stem}.json"
        data = {
            "document_name": pdf_path.stem,
            "pages": []
        }

        for page_number, content in page_contents.items():
            page_data = {
                "page_number": page_number,
                "content": json.loads(content) # Parse the json string back to object
            }
            data["pages"].append(page_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)