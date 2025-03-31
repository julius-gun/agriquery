from pathlib import Path
from typing import Dict
import re
import xml.etree.ElementTree as ET  # Import ElementTree for HTML parsing
from .base_converter import BaseConverter
from bs4 import BeautifulSoup


class HtmlConverter(BaseConverter):
    def convert_to_format(self, doc, pdf_path: Path, output_dir: Path) -> Dict[int, str]:
        """
        Converts the document to HTML format, extracting only the body content.
        """
        pages_content = {}
        for page_number in doc.pages.keys():
            full_html_content = doc.export_to_html(page_no=page_number)
            body_content = self._extract_body_content(full_html_content) # Extract body content
            pages_content[page_number] = body_content
        return pages_content

    def _extract_body_content(self, full_html: str) -> str:
        """
        Extracts the content within the <body> tags from a full HTML string.
        Also removes unnecessary nested <ul> tags that are empty or contain only other empty <ul> tags.
        """
        try:
            root = ET.fromstring(full_html)
            body = root.find('body')
            if body is not None:
                # Remove unnecessary nested <ul> tags
                for ul in body.findall('.//ul'):
                    self._remove_empty_nested_ul(ul)

                body_content_str = ''.join(ET.tostring(element, encoding='unicode') for element in body)
                body_content_str = body_content_str.replace('<body>', '', 1).replace('</body>', '', 1).strip()
                print("body_content_str: ", body_content_str)
                return body_content_str
            else:
                return full_html
        except ET.ParseError:
            return full_html

    def _remove_empty_nested_ul(self, ul_element):
        """
        Recursively removes nested <ul> tags that are empty or contain only other empty <ul> tags.
        """
        for child_ul in ul_element.findall('.//ul'):
            self._remove_empty_nested_ul(child_ul)  # Recursively process child <ul> elements

        # Check if the current <ul> element is empty or contains only whitespace
        if not ul_element.text and not ul_element.tail and not any(child.tag != 'ul' or child.text or child.tail for child in ul_element):
            parent = ul_element.getparent()
            if parent is not None:
                parent.remove(ul_element)

    def save_with_original_extension(self, page_contents: Dict[int, str], pdf_path: Path, output_dir: Path, doc):
        """
        Saves the converted content to HTML files (.html).  Saves full HTML, not just body.
        """
        output_path = output_dir / f"{pdf_path.stem}.html"
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>PDF Conversion</title>",
            "<style>",
            "body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif; box-sizing: border-box; }",
            "img { max-width: 100%; height: auto; }",
            ".page { margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; box-sizing: border-box; }",
            "h2 { color: #333; }",
            ".page ul {",
            "    max-width: 100%;",
            "    box-sizing: border-box;",
            "    padding-left: 1px;",
            "}",
            ".page li {",
            "    word-wrap: break-word;",
            "    overflow-wrap: break-word;",
            "}",
            "</style>",
            "</head>",
            "<body>",
        ]

        for page_number, content in page_contents.items():
            html_content.append(f'<div class="page">')
            html_content.append(f"<h2>Page {page_number}</h2>")
            html_content.append(content)
            html_content.append("</div>")

        html_content.extend(["</body>", "</html>"])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))
