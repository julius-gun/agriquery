<p align="center">
  <img src="assets/docu_llama.jpg" alt="docu_llama" width="300"/>
  <br>
</p>

# Docling page-wise PDF converter
Page-wise PDF converter built with Docling, offering modular and extensible conversion to various formats.

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Description:**

`Docling-page-wise-pdf-converter` is a Python package designed to convert PDF documents into various formats on a page-by-page basis. It leverages the `docling` library for document parsing and conversion, providing a modular and extensible way to extract content from PDFs and save it in formats like **Markdown, HTML, Plain Text, JSON, YAML, CSV, and XML**.

**Key Features:**

*   **Page-wise Conversion:** Converts PDF content page by page, allowing for granular access and processing.
*   **Multiple Output Formats:** Supports conversion to Markdown, HTML, TXT, JSON, YAML, CSV, and XML.
*   **Content Extraction:** Extracts text, tables, and image captions from PDF documents.
*   **Format-Specific Output:** Saves content in structured formats suitable for different use cases (e.g., Markdown for readability, JSON/YAML for data processing).
*   **Efficient Conversion Management:** Avoids redundant conversions by checking if a format has already been generated.
*   **Extensible Architecture:** Easily add support for new output formats by implementing new converter classes.
*   **Clean Code and Modular Design:** Follows clean code principles with well-separated modules for content management, format conversion, and core PDF processing.
*   **Graphical User Interface (GUI):** Includes a simple GUI for easy file and directory selection.

**File Structure:**


- [`docling-page-wise-pdf-converter`]()
  - [`__init__.py`](__init__.py)
  - [`content_manager.py`](content_manager.py)
  - [`format_converters/`](format_converters/)
    - [`__init__.py`](format_converters/__init__.py)
    - [`base_converter.py`](format_converters/base_converter.py)
    - [`csv_converter.py`](format_converters/csv_converter.py)
    - [`html_converter.py`](format_converters/html_converter.py)
    - [`json_converter.py`](format_converters/json_converter.py)
    - [`markdown_converter.py`](format_converters/markdown_converter.py)
    - [`txt_converter.py`](format_converters/txt_converter.py)
    - [`xml_converter.py`](format_converters/xml_converter.py)
    - [`yaml_converter.py`](format_converters/yaml_converter.py)
  - [`pdf_converter.py`](pdf_converter.py)
  - [`interactive_pdf_converter.py`](interactive_pdf_converter.py)  <- **NEW: GUI Script**
  - [`utils.py`](utils.py)
  - [`README.md`](README.md)

**Usage:**

1.  **Installation (Prerequisites):**

    Ensure you have Python 3.9+ installed. You need to install the `docling` and `docling-core` libraries:


    **Using `requirements.txt` (Recommended):**

    Navigate to the root directory of the `Docling-page-wise-pdf-converter` package in your terminal and run:

    ```bash
    pip install -r requirements.txt
    ```

    This will install `docling`, `docling-core`, `PyYAML`, and `pydantic` along with their dependencies.

2.  **Using the GUI (`pdf_converter_gui.py`):**

    For users who prefer a graphical interface, a simple GUI script `pdf_converter_gui.py` is provided.

    **To run the GUI:**

    Navigate to the root directory of the `docling_page_wise_pdf_converter` in your terminal and run:

    ```bash
    python interactive_pdf_converter.py
    ```

    The script will:

    *   Open a file dialog prompting you to select one or more PDF files for conversion.
    *   After selecting files, it will open a directory dialog asking you to choose an output directory.
    *   Once you've selected the output directory, the conversion process will start for each selected PDF file, converting them to all supported formats and saving the output in the chosen directory.
    *   Conversion progress and completion will be logged in the console.
    

3.  **Using the `convert_pdf` function:**

    The main entry point for conversion is the `convert_pdf` function in `pdf_converter.py`.

    ```python
    from docling_page_wise_pdf_converter.pdf_converter import convert_pdf

    pdf_file = "path/to/your/pdf_file.pdf"  # Replace with your PDF file path
    output_directory = "output_folder"  # Replace with desired output directory

    # Convert to all formats
    convert_pdf(pdf_file, output_directory, output_format="all")

    # Convert to a specific format (e.g., Markdown)
    convert_pdf(pdf_file, output_directory, output_format="markdown")

    # Supported output formats: "markdown", "html", "txt", "json", "yaml", "csv", "xml", "all"
    ```

    The converted files and an `images` folder (containing extracted images) will be saved in the `output_directory`.

4.  **Getting Page Content in Plain Text:**

    You can retrieve the plain text content of specific pages after conversion using the `get_page_content` method of the `PdfConverter` class, or directly using the `ContentManager`.

    **Example (using `ContentManager` after `convert_pdf` has been run):**

    ```python

    pdf_stem = Path(pdf_file).stem # Get the PDF filename without extension
    content_manager = ContentManager(Path(output_directory))

    # Get plain text content of page 1 from TXT format
    plain_text_page_1 = content_manager.get_page_content_plain_text(pdf_stem, "txt", 1)
    if plain_text_page_1:
        print(f"Page 1 content (TXT):\n{plain_text_page_1[:500]}...")

    # Get plain text content of pages 4, 10, and 11 from HTML format
    plain_text_pages_4_10_11 = content_manager.get_page_content_plain_text(pdf_stem, "html", [4, 10, 11])
    if plain_text_pages_4_10_11:
        print(f"Pages 4, 10, 11 content (HTML):")
        for i, content in enumerate(plain_text_pages_4_10_11):
            print(f"Page { [4, 10, 11][i] } content:\n{content[:200]}...")
    ```