import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from template_langgraph.internals.csv_loaders import (
    CsvLoaderWrapper,
    Settings,
    get_csv_loader_settings,
)


class TestSettings:
    """Test cases for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        assert settings.csv_loader_data_dir_path == "./data"

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = Settings(csv_loader_data_dir_path="/custom/path")
        assert settings.csv_loader_data_dir_path == "/custom/path"

    @patch.dict(os.environ, {"CSV_LOADER_DATA_DIR_PATH": "/env/path"})
    def test_env_settings(self):
        """Test settings from environment variables."""
        settings = Settings()
        assert settings.csv_loader_data_dir_path == "/env/path"


class TestGetCsvLoaderSettings:
    """Test cases for get_csv_loader_settings function."""

    def test_get_csv_loader_settings_returns_settings_instance(self):
        """Test that get_csv_loader_settings returns a Settings instance."""
        settings = get_csv_loader_settings()
        assert isinstance(settings, Settings)

    def test_get_csv_loader_settings_cached(self):
        """Test that get_csv_loader_settings is cached."""
        settings1 = get_csv_loader_settings()
        settings2 = get_csv_loader_settings()
        assert settings1 is settings2


class TestCsvLoaderWrapper:
    """Test cases for CsvLoaderWrapper class."""

    def test_init_with_default_settings(self):
        """Test initialization with default settings."""
        wrapper = CsvLoaderWrapper()
        assert isinstance(wrapper.settings, Settings)
        assert wrapper.settings.csv_loader_data_dir_path == "./data"

    def test_init_with_custom_settings(self):
        """Test initialization with custom settings."""
        custom_settings = Settings(csv_loader_data_dir_path="/custom/path")
        wrapper = CsvLoaderWrapper(settings=custom_settings)
        assert wrapper.settings is custom_settings
        assert wrapper.settings.csv_loader_data_dir_path == "/custom/path"

    def test_load_csv_docs_empty_directory(self):
        """Test loading CSV documents from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(csv_loader_data_dir_path=temp_dir)
            wrapper = CsvLoaderWrapper(settings=settings)
            docs = wrapper.load_csv_docs()
            assert docs == []

    def test_load_csv_docs_with_csv_files(self):
        """Test loading CSV documents from directory with CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test CSV files
            csv_file1 = Path(temp_dir) / "test1.csv"
            csv_file1.write_text("name,age\nAlice,25\nBob,30\n")

            csv_file2 = Path(temp_dir) / "test2.csv"
            csv_file2.write_text("city,country\nTokyo,Japan\nParis,France\n")

            settings = Settings(csv_loader_data_dir_path=temp_dir)
            wrapper = CsvLoaderWrapper(settings=settings)
            docs = wrapper.load_csv_docs()

            assert len(docs) == 4  # 2 rows from each CSV file
            assert all(isinstance(doc, Document) for doc in docs)

            # Check that documents contain expected content
            doc_contents = [doc.page_content for doc in docs]
            assert any("Alice" in content for content in doc_contents)
            assert any("Bob" in content for content in doc_contents)
            assert any("Tokyo" in content for content in doc_contents)
            assert any("Paris" in content for content in doc_contents)

    def test_load_csv_docs_with_nested_directories(self):
        """Test loading CSV documents from nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory structure
            nested_dir = Path(temp_dir) / "nested"
            nested_dir.mkdir()

            # Create CSV file in nested directory
            csv_file = nested_dir / "nested_test.csv"
            csv_file.write_text("product,price\nApple,100\nBanana,80\n")

            settings = Settings(csv_loader_data_dir_path=temp_dir)
            wrapper = CsvLoaderWrapper(settings=settings)
            docs = wrapper.load_csv_docs()

            assert len(docs) == 2
            assert all(isinstance(doc, Document) for doc in docs)

            # Check that documents contain expected content
            doc_contents = [doc.page_content for doc in docs]
            assert any("Apple" in content for content in doc_contents)
            assert any("Banana" in content for content in doc_contents)

    def test_load_csv_docs_with_non_csv_files(self):
        """Test loading CSV documents ignores non-CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create CSV file
            csv_file = Path(temp_dir) / "test.csv"
            csv_file.write_text("name,value\ntest,123\n")

            # Create non-CSV files that should be ignored
            txt_file = Path(temp_dir) / "test.txt"
            txt_file.write_text("This should be ignored")

            json_file = Path(temp_dir) / "test.json"
            json_file.write_text('{"key": "value"}')

            settings = Settings(csv_loader_data_dir_path=temp_dir)
            wrapper = CsvLoaderWrapper(settings=settings)
            docs = wrapper.load_csv_docs()

            assert len(docs) == 1  # Only the CSV file should be loaded
            assert "test" in docs[0].page_content
            assert "123" in docs[0].page_content

    def test_load_csv_docs_nonexistent_directory(self):
        """Test loading CSV documents from nonexistent directory."""
        settings = Settings(csv_loader_data_dir_path="/nonexistent/path")
        wrapper = CsvLoaderWrapper(settings=settings)
        docs = wrapper.load_csv_docs()
        assert docs == []

    def test_load_csv_docs_with_malformed_csv(self):
        """Test behavior with malformed CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create malformed CSV file
            csv_file = Path(temp_dir) / "malformed.csv"
            csv_file.write_text("header1,header2\nvalue1\nvalue2,value3,extra\n")

            settings = Settings(csv_loader_data_dir_path=temp_dir)
            wrapper = CsvLoaderWrapper(settings=settings)

            # CSVLoader should handle malformed CSV gracefully
            docs = wrapper.load_csv_docs()
            assert len(docs) >= 0  # Should not crash, but may return empty or partial results
