import os
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from template_langgraph.internals.pdf_loaders import (
    PdfLoaderWrapper,
    Settings,
    get_pdf_loader_settings,
)


class TestSettings:
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        assert settings.pdf_loader_data_dir_path == "./data"

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = Settings(pdf_loader_data_dir_path="/custom/path")
        assert settings.pdf_loader_data_dir_path == "/custom/path"

    @patch.dict(os.environ, {"PDF_LOADER_DATA_DIR_PATH": "/env/path"})
    def test_env_settings(self):
        """Test settings from environment variables."""
        settings = Settings()
        assert settings.pdf_loader_data_dir_path == "/env/path"


class TestGetPdfLoaderSettings:
    def test_get_pdf_loader_settings_returns_settings(self):
        """Test that get_pdf_loader_settings returns a Settings instance."""
        settings = get_pdf_loader_settings()
        assert isinstance(settings, Settings)

    def test_get_pdf_loader_settings_is_cached(self):
        """Test that get_pdf_loader_settings is cached."""
        settings1 = get_pdf_loader_settings()
        settings2 = get_pdf_loader_settings()
        assert settings1 is settings2


class TestPdfLoaderWrapper:
    def test_init_with_default_settings(self):
        """Test PdfLoaderWrapper initialization with default settings."""
        wrapper = PdfLoaderWrapper()
        assert isinstance(wrapper.settings, Settings)
        assert wrapper.settings.pdf_loader_data_dir_path == "./data"

    def test_init_with_custom_settings(self):
        """Test PdfLoaderWrapper initialization with custom settings."""
        custom_settings = Settings(pdf_loader_data_dir_path="/custom/path")
        wrapper = PdfLoaderWrapper(settings=custom_settings)
        assert wrapper.settings.pdf_loader_data_dir_path == "/custom/path"

    @patch("template_langgraph.internals.pdf_loaders.glob")
    @patch("template_langgraph.internals.pdf_loaders.PyPDFLoader")
    def test_load_pdf_docs_no_files(self, mock_pdf_loader, mock_glob):
        """Test load_pdf_docs when no PDF files are found."""
        mock_glob.return_value = []
        wrapper = PdfLoaderWrapper()

        docs = wrapper.load_pdf_docs()

        assert docs == []
        mock_glob.assert_called_once_with(
            os.path.join("./data", "**", "*.pdf"),
            recursive=True,
        )

    @patch("template_langgraph.internals.pdf_loaders.glob")
    @patch("template_langgraph.internals.pdf_loaders.PyPDFLoader")
    def test_load_pdf_docs_with_files(self, mock_pdf_loader, mock_glob):
        """Test load_pdf_docs when PDF files are found."""
        # Setup mock data
        mock_glob.return_value = ["./data/file1.pdf", "./data/file2.pdf"]

        mock_doc1 = Document(page_content="Content 1", metadata={"source": "file1.pdf"})
        mock_doc2 = Document(page_content="Content 2", metadata={"source": "file1.pdf"})
        mock_doc3 = Document(page_content="Content 3", metadata={"source": "file2.pdf"})

        mock_loader_instance1 = Mock()
        mock_loader_instance1.load_and_split.return_value = [mock_doc1, mock_doc2]

        mock_loader_instance2 = Mock()
        mock_loader_instance2.load_and_split.return_value = [mock_doc3]

        mock_pdf_loader.side_effect = [mock_loader_instance1, mock_loader_instance2]

        wrapper = PdfLoaderWrapper()
        docs = wrapper.load_pdf_docs()

        # Verify results
        assert len(docs) == 3
        assert docs[0] == mock_doc1
        assert docs[1] == mock_doc2
        assert docs[2] == mock_doc3

        # Verify PyPDFLoader was called correctly
        assert mock_pdf_loader.call_count == 2
        mock_pdf_loader.assert_any_call("./data/file1.pdf")
        mock_pdf_loader.assert_any_call("./data/file2.pdf")

        # Verify load_and_split was called with text splitter
        mock_loader_instance1.load_and_split.assert_called_once()
        mock_loader_instance2.load_and_split.assert_called_once()

    @patch("template_langgraph.internals.pdf_loaders.glob")
    @patch("template_langgraph.internals.pdf_loaders.PyPDFLoader")
    def test_load_pdf_docs_with_custom_data_dir(self, mock_pdf_loader, mock_glob):
        """Test load_pdf_docs with custom data directory."""
        custom_settings = Settings(pdf_loader_data_dir_path="/custom/data")
        wrapper = PdfLoaderWrapper(settings=custom_settings)
        mock_glob.return_value = []

        wrapper.load_pdf_docs()

        mock_glob.assert_called_once_with(
            os.path.join("/custom/data", "**", "*.pdf"),
            recursive=True,
        )

    @patch("template_langgraph.internals.pdf_loaders.glob")
    @patch("template_langgraph.internals.pdf_loaders.PyPDFLoader")
    def test_load_pdf_docs_text_splitter_configuration(self, mock_pdf_loader, mock_glob):
        """Test that text splitter is configured correctly."""
        mock_glob.return_value = ["./data/test.pdf"]
        mock_loader_instance = Mock()
        mock_loader_instance.load_and_split.return_value = []
        mock_pdf_loader.return_value = mock_loader_instance

        wrapper = PdfLoaderWrapper()
        wrapper.load_pdf_docs()

        # Verify that load_and_split was called with a text splitter
        mock_loader_instance.load_and_split.assert_called_once()
        args = mock_loader_instance.load_and_split.call_args[0]
        assert len(args) == 1
        text_splitter = args[0]

        # Verify text splitter configuration
        assert hasattr(text_splitter, "_chunk_size")
        assert hasattr(text_splitter, "_chunk_overlap")
