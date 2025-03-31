# RAG/document_processor.py
from llama_index.core import Document
from langchain_text_splitters import TokenTextSplitter

class DocumentProcessor:
    """Processes documents into nodes for RAG."""

    def __init__(self, rag_config):
        self.rag_config = rag_config
        self.chunk_size = rag_config.get("chunk_size", 1024)
        self.node_parsers = {
            ".txt": TokenTextSplitter(chunk_size=self.chunk_size), # Default parser for other text-based formats
        }
        self.default_parser = TokenTextSplitter(chunk_size=self.chunk_size)


    def process_document(self, pages):
        """Processes a list of document pages into nodes."""
        nodes = []
        for page in pages:
            file_extension = page.metadata.get('file_extension', '.txt') # Default to .txt if extension not found
            parser = self.node_parsers.get(file_extension, self.default_parser) # Use default parser if no specific parser found
            document = Document(text=page.text, metadata=page.metadata)
            page_nodes = parser.get_nodes_from_documents([document])
            nodes.extend(page_nodes)
        return nodes