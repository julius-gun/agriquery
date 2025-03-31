

# python -m document_loaders.document_loader
import logging  # Import the logging module

from pathlib import Path
from docling_page_wise_pdf_converter.pdf_converter import convert_pdf
from docling_page_wise_pdf_converter.content_manager import ContentManager
from typing import List, Dict
from utils.config_loader import ConfigLoader
import os
import requests  # Import requests

# Configure logging (basic setup - you might want a more sophisticated setup)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents of various formats and extracts page-wise content."""

    def __init__(self, output_directory: str = "results", config_path: str = "config.json", local_filename: str = None):  # Add config_path and local_filename
        self.output_directory = Path(output_directory)
        self.converted_documents_dir  = self.output_directory / "converted_documents" # Subdirectory for converted docs
        self.content_manager = ContentManager(self.converted_documents_dir)
        self.local_filename = local_filename if local_filename else "document" # Default local filename
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.converted_documents_dir.mkdir(parents=True, exist_ok=True) # Ensure converted_documents_dir exists
        self.config_loader = ConfigLoader(config_path) # Load config
        self.file_extensions_to_test = self.config_loader.config.get("file_extensions_to_test", ["txt"]) # Get extensions from config, else default to ["txt"]

    def load_document(self, file_path: str) -> List[Dict[str, str]]:
        """
        Loads a document and returns page-wise content in plain text format.

        Args:
            file_path (str): Path to the document file.

        Returns:
            List[Dict[str, str]]: List of dictionaries, where each dictionary represents a page
                                  and contains 'page' number and 'content' (plain text).
        """
        file_path_p = Path(file_path) 
        file_extension = file_path_p.suffix.lower()
        pdf_stem = self.local_filename # Use local filename as stem
        output_txt_path = self.converted_documents_dir / f"{pdf_stem}.txt.json"

        if not file_path.startswith("http"): # Local file
            raise ValueError("Only URLs are supported in this version.")


        # Convert PDF if it hasn't been converted yet
        if not output_txt_path.exists():
            if file_extension != ".pdf":
                logger.error(f"Unsupported file format: {file_extension}")
                raise ValueError(f"Unsupported file format: {file_extension}")
            # Download the PDF from the URL
            try:
                response = requests.get(file_path, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes

                temp_pdf_path = self.converted_documents_dir / "temp_downloaded.pdf"
                with open(temp_pdf_path, "wb") as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)
                logger.info(f"PDF downloaded to {temp_pdf_path}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading PDF from {file_path}: {e}")
                raise
            logger.info(f"Converting PDF '{temp_pdf_path}' to other formats in '{self.converted_documents_dir}'.")
            convert_pdf(source=str(temp_pdf_path), output_dir=str(self.converted_documents_dir), output_format="all")

            # --- Renaming ALL files ---
            original_stem = Path(temp_pdf_path).stem  # Get "temp_downloaded"
            for converted_file in self.converted_documents_dir.iterdir():
                if converted_file.name.startswith(original_stem):
                    new_filepath = self.converted_documents_dir / converted_file.name.replace(original_stem, pdf_stem)
                    os.rename(converted_file, new_filepath)
                    logger.info(f"Renamed '{converted_file}' to '{new_filepath}'")


        elif output_txt_path.exists():
            logger.info(f"PDF '{file_path}' already converted. Skipping conversion.")

        pages_content = []
        for ext in self.file_extensions_to_test:
            logger.info(f"Attempting to load content for extension: {ext}")  # Log extension
            content = self.content_manager.load_content(pdf_stem, ext)
            if content:
                logger.info(f"Successfully loaded content for extension: {ext}")  # Log success
                for page_num_str, page_content in content.items():
                    try:
                        page_num = int(page_num_str)  # Ensure page_num is an integer
                    except ValueError:
                        logger.warning(f"Invalid page number '{page_num_str}'. Skipping.")
                        continue
                    pages_content.append({"page": f"Page {page_num}", "content": page_content, "file_extension": ext})
            else:
                logger.warning(f"Failed to load content for extension: {ext}")

        return pages_content


if __name__ == '__main__':
    # Example usage:
    local_filename = "english_manual"
    pdf_url = "https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703540-2.pdf"
    document_loader = DocumentLoader(local_filename=local_filename)  # Pass local_filename
    try:
        pages = document_loader.load_document(pdf_url)
        if pages:
            print(f"Loaded {len(pages)} pages from {pdf_url}")
            first_page_txt = next((page for page in pages if page['page'] == 'Page 1' and page['file_extension'] == "txt"), None)
            if first_page_txt:
                print(f"Page 1 Content (.txt):\n{first_page_txt['content'][:500]}...")
            else:
                print("Page 1 content in .txt format not found.")
        else:
            print(f"No pages loaded from {pdf_url}")
    except Exception as e:
        print(f"Error loading document: {e}")