# docling-page-wise-pdf-converter/format_converters/csv_converter.py
import csv
from pathlib import Path
from typing import Dict, List
from docling_core.types.doc import TableItem, PictureItem, TextItem
from .base_converter import BaseConverter

class CsvConverter(BaseConverter):
    def convert_to_format(self, doc, pdf_path: Path, output_dir: Path) -> Dict[int, List[Dict]]: # Return type is now Dict[int, List[Dict]]
        """
        Converts the document to CSV format and returns page-based data.
        CSV export data is prepared here and saved in save_with_original_extension.
        Returns a dictionary where keys are page numbers and values are lists of CSV rows (dictionaries).
        """
        pages_csv_data: Dict[int, List[Dict]] = {} # Initialize a dictionary to hold page-based CSV data

        for page_number in doc.pages.keys():
            csv_rows_for_page: List[Dict] = [] # List to hold CSV rows for the current page
            for item, _ in doc.iterate_items(page_no=page_number):
                try:
                    element_type = item.__class__.__name__
                    content = ""
                    additional_info = ""

                    if isinstance(item, TextItem):
                        content = item.text
                    elif isinstance(item, TableItem):
                        try:
                            content = "Table"
                            additional_info = str(item.export_to_dataframe())

                        except:
                            content = "Table (not extractable)"
                    elif isinstance(item, PictureItem):
                        content = "Image"
                        additional_info = item.caption_text(doc) if hasattr(item, 'caption_text') else ""

                    csv_rows_for_page.append({ # Append a dictionary representing a CSV row
                        "page_number": page_number,
                        "element_type": element_type,
                        "content": content,
                        "additional_info": additional_info
                    })
                except Exception as e:
                    print(f"Warning: Failed to process item for CSV on page {page_number}: {str(e)}")
            pages_csv_data[page_number] = csv_rows_for_page # Store the list of rows for the current page

        return pages_csv_data # Return the page-based CSV data

    def save_with_original_extension(self, page_contents: Dict[int, List[Dict]], pdf_path: Path, output_dir: Path, doc):
        """
        Saves the converted content to csv files (.csv).
        'page_contents' here is a dictionary where keys are page numbers and values are lists of CSV rows (dictionaries).
        """
        output_path = output_dir / f"{pdf_path.stem}.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["page_number", "element_type", "content", "additional_info"]) # Write header

            for page_number, csv_rows in page_contents.items(): # Iterate through page-based CSV data
                for row_dict in csv_rows: # Iterate through rows for each page
                    writer.writerow([
                        row_dict["page_number"],
                        row_dict["element_type"],
                        row_dict["content"],
                        row_dict["additional_info"]
                    ])