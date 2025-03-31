from pathlib import Path
from typing import Dict, List, Optional
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem

import sys
import os

# Change relative imports to absolute imports
from .content_manager import ContentManager
from .format_converters.markdown_converter import MarkdownConverter
from .format_converters.html_converter import HtmlConverter
from .format_converters.txt_converter import TxtConverter
from .format_converters.json_converter import JsonConverter
from .format_converters.yaml_converter import YamlConverter
from .format_converters.csv_converter import CsvConverter
from .format_converters.xml_converter import XmlConverter


class PdfConverter:
    """
    Converts PDF documents to various formats.
    """
    def __init__(self, source: str, output_dir: str):
        self.source = source
        self.output_dir = Path(output_dir)
        # Create a filename from URL or use local path
        if '://' in source:
            self.output_filename = Path(source.split('/')[-1])
            if not self.output_filename.suffix:
                self.output_filename = Path(f"{self.output_filename}.pdf")
        else:
            self.output_filename = Path(source)
        
        self.pdf_stem = self.output_filename.stem
        self.images_dir = self.output_dir / "images"
        self.content_manager = ContentManager(self.output_dir)
        self.converter = self._initialize_converter()
        # Direct conversion from source (works with both URLs and local files)
        self.result = self.converter.convert(source)
        self.doc = self.result.document
        self.format_converters = {
            "markdown": MarkdownConverter(),
            "html": HtmlConverter(),
            "txt": TxtConverter(),
            "json": JsonConverter(),
            "yaml": YamlConverter(),
            "csv": CsvConverter(),
            "xml": XmlConverter(),
        }

    def _initialize_converter(self):
        """
        Initializes the DocumentConverter with PDF pipeline options.
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _convert_and_save_format(self, format_name: str):
        """
        Converts the PDF to the specified format and saves the content.
        """
        if self.content_manager.has_content(self.pdf_stem, format_name):
            print(f"Content for {format_name} already exists. Skipping conversion.")
            return

        if format_name not in self.format_converters:
            raise ValueError(f"Unsupported output format: {format_name}")

        converter = self.format_converters[format_name]
        page_contents = converter.convert_to_format(self.doc, self.output_filename, self.output_dir)
        self.content_manager.save_content(self.pdf_stem, format_name, page_contents)

        # Save with original extension if applicable and desired (e.g., for markdown, html, txt, xml, csv, yaml)
        if format_name in ["markdown", "html", "txt", "json", "yaml", "csv", "xml"]:
            converter.save_with_original_extension(page_contents, self.output_filename, self.output_dir, self.doc) # Pass self.doc here

    def export_images(self) -> List[Path]:
        """Exports images from the document."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        try:
            doc_filename = self.source.stem
            image_paths = []

            # Export page images
            for page_no, page in self.doc.pages.items():
                try:
                    if hasattr(page, 'image') and page.image and hasattr(page.image, 'pil_image'):
                        image_path = self.images_dir / f"{doc_filename}_page_{page_no}.png"
                        page.image.pil_image.save(image_path, format="PNG")
                        image_paths.append(image_path)
                except Exception as e:
                    print(f"Warning: Failed to save page {page_no} image: {str(e)}")

            # Export figures and tables
            table_counter = picture_counter = 0

            for element, _ in self.doc.iterate_items():
                try:
                    if isinstance(element, TableItem) and hasattr(element, 'get_image'):
                        table_counter += 1
                        page_no = element.prov[0].page_no if element.prov else 0
                        image_path = self.images_dir / f"{doc_filename}_page_{page_no}_table_{table_counter}.png"
                        table_image = element.get_image(self.doc)
                        if table_image:
                            table_image.save(image_path, "PNG")
                            image_paths.append(image_path)

                    if isinstance(element, PictureItem) and hasattr(element, 'get_image'):
                        picture_counter += 1
                        page_no = element.prov[0].page_no if element.prov else 0
                        image_path = self.images_dir / f"{doc_filename}_page_{page_no}_picture_{picture_counter}.png"
                        picture_image = element.get_image(self.doc)
                        if picture_image:
                            picture_image.save(image_path, "PNG")
                            image_paths.append(image_path)
                except Exception as e:
                    print(f"Warning: Failed to save element image: {str(e)}")

            return image_paths
        except Exception as e:
            print(f"Warning: Image export partially failed: {str(e)}")
            return []


    def convert_all(self):
        """Converts PDF to all supported formats and exports images."""
        # self.export_images()
        for format_name in self.format_converters:
            self._convert_and_save_format(format_name)

    def convert_to_format(self, output_format: str):
        """Converts PDF to the specified format and exports images."""
        # self.export_images() # Commented out to avoid exporting images multiple times
        self._convert_and_save_format(output_format)

    def get_page_content(self, output_format: str, page: int) -> Optional[str]:
        """
        Retrieves the page content in plain text for a specific format and page number.
        """
        return self.content_manager.get_page_content_plain_text(self.pdf_stem, output_format, page)


def convert_pdf(source: str, output_dir: str, output_format: str = "all"):
    """
    Converts PDF to multiple formats and export images.
    Args:
        source: Path to the input PDF file or URL
        output_dir: Directory for output files
        output_format: The desired output format (e.g., "markdown", "html", "txt", "json", "yaml", "csv", "xml", or "all").
                       Defaults to "all".
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    converter = PdfConverter(source, output_dir)
    if output_format == "all":
        converter.convert_all()
    else:
        converter.convert_to_format(output_format)


# Example usage:
if __name__ == "__main__":
    pdf_file = "https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703540-2.pdf"  # Replace with your PDF file path
    
    output_directory = f"output/{Path(pdf_file).stem}/"
    convert_pdf(pdf_file, output_directory, output_format="all")

    # Example of getting page content
    pdf_stem = Path(pdf_file).stem # Use stem to match how it's saved internally
    content_manager = ContentManager(Path(output_directory))
    plain_text_content = content_manager.get_page_content_plain_text(pdf_stem, "txt", [4,10,11])
    # plain_text_content = content_manager.get_page_content_plain_text(pdf_stem, "txt", 4)
    print(plain_text_content)

    # create a markdown table showing the content of the pdf of page 1 in all formats            
    print("\n\n\n")
    print("| Format | Content |")
    print("| --- | --- |")
    for format_name in ["markdown", "html", "txt", "json", "yaml", "csv", "xml"]:
        content = content_manager.get_page_content_plain_text(pdf_stem, format_name, 1)
        if content:
            print(f"| {format_name.upper()} | {content[:50]}... |") 
        else:
            print(f"| {format_name.upper()} | Could not retrieve content |")
    print("\n\n\n")