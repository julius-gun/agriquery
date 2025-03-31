# docling-page-wise-pdf-converter/content_manager.py
import json
from pathlib import Path
from typing import Dict, Optional, Union, List
import logging  # Import logging

# Configure logging (if not already configured elsewhere)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContentManager:
    """
    Manages the storage and retrieval of converted content for PDF documents.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_content_path(self, pdf_stem: str, format_name: str) -> Path:
        """
        Constructs the path to the content file for a given format.
        """
        return self.output_dir / f"{pdf_stem}.{format_name}.json"

    def has_content(self, pdf_stem: str, format_name: str) -> bool:
        """
        Checks if content for a given format already exists.
        """
        content_path = self._get_content_path(pdf_stem, format_name)
        return content_path.exists()

    def save_content(self, pdf_stem: str, format_name: str, page_contents: Dict[int, str]):
        """
        Saves the page content to a JSON file.
        """
        content_path = self._get_content_path(pdf_stem, format_name)
        data = []
        for page_num, content in page_contents.items():
            data.append({"page": page_num, "content": content})
        with open(content_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_content(self, pdf_stem: str, format_name: str) -> Optional[Dict[int, str]]:
        """
        Loads the page content from a JSON file.
        """
        content_path = self._get_content_path(pdf_stem, format_name)
        if not content_path.exists():
            return None
        page_contents = {}
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    page_contents[item['page']] = item['content']
            return page_contents
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {content_path}. File might be corrupted.")
            return None

    def get_page_content_plain_text(self, pdf_stem: str, format_name: str, page: Union[int, List[int]]) -> Optional[Union[str, List[str]]]:
        """
        Retrieves the plain text content of a specific page or pages from a saved format.

        Args:
            pdf_stem: Stem of the PDF file name.
            format_name: Format name (e.g., "txt", "html").
            page: A single page number (int) or a list of page numbers (List[int]).

        Returns:
            If page is int: Plain text content of the specified page, or None if not found.
            If page is List[int]: A list of plain text contents for the specified pages,
                                  or None if content for any page is not found.
        """
        page_contents = self.load_content(pdf_stem, format_name)
        if not page_contents:
            return None

        if isinstance(page, int):
            if page in page_contents:
                return page_contents[page]
            else:
                return None
        elif isinstance(page, list):
            results = ""
            for page_num in page:
                if page_num in page_contents:
                    results += page_contents[page_num]
                else:
                    return None  # Return None if content for any page in the list is missing
            return results.strip()
        else:
            raise TypeError("page must be an int or a list of ints")