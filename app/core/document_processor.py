"""Document Processing Module for Loading and Chunking documents."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor():
    """Process the documents for RAG pipeline."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, chunk_size:int|None = None, chunk_overlap:int|None = None) -> None:
        """ Initialize Document Processor.
        
        Args:
            chunk_size : size of text chunks(default from settings)
            chunk_overlap : overlap between chunks(default from settings)
        """

        settings = get_settings
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=['\n','\n\n','.',' ','']
        )

        logger.info(f"Document Processor initialized with chunk size = {chunk_overlap}, "
                    f"chunk overlap = {chunk_overlap}"
        )

    def load_pdf(self, file_path:str|Path=None) -> list[Document]:
        """Load a PDF File. (1. path 2. load)

        Args:
            file_path: Path to PDF file.

        Returns:
            List of Document objects
        """

        file_path = Path(file_path)
        logger.debug(f"Loading pdf : {file_path.name}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} pages from {file_path.name}.")
        return documents

    def load_file(self, file_path:str|Path) -> list[Document]:
        """Load a file based on its extension.

        Args:
            file_path: Path to File

        Returns:
            List of Document objects

        Raises:
            ValueError: If file extension is not supported.
        """

        file_path = Path(file_path)
        extension_of_file = file_path.suffix.lower()

        if extension_of_file not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension : {extension_of_file}."
                f"Supported : {self.SUPPORTED_EXTENSIONS}"
            )

        loaders = {
            ".pdf" : self.load_pdf
        }

        return loaders[extension_of_file](file_path)

    
