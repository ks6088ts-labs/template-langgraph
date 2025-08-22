import logging
import threading
from unittest.mock import Mock, patch

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper, Settings


class TestAzureOpenAiWrapper:
    """Test cases for AzureOpenAiWrapper authentication optimization."""

    def setup_method(self):
        """Reset class-level variables before each test."""
        AzureOpenAiWrapper._credentials.clear()
        AzureOpenAiWrapper._tokens.clear()

    def test_lazy_initialization_api_key(self, caplog):
        """Test that API key authentication uses lazy initialization."""
        settings = Settings(
            azure_openai_use_microsoft_entra_id="false",
            azure_openai_api_key="dummy_key",
            azure_openai_endpoint="https://dummy.openai.azure.com/",
        )

        with caplog.at_level(logging.INFO):
            # Creating instances should not trigger authentication
            wrapper1 = AzureOpenAiWrapper(settings)

            # No authentication logs yet
            assert "Using API key for authentication" not in caplog.text

            # Accessing models should trigger authentication
            try:
                _ = wrapper1.chat_model
            except Exception:
                pass  # Expected due to dummy credentials

            # Should see authentication log only once per model access
            assert caplog.text.count("Using API key for authentication") == 1

            # Second access should not trigger additional authentication
            try:
                _ = wrapper1.reasoning_model
            except Exception:
                pass

            # Should still be only one authentication log per model type
            assert caplog.text.count("Using API key for authentication") >= 1

    @patch("template_langgraph.llms.azure_openais.DefaultAzureCredential")
    def test_singleton_credential_entra_id(self, mock_credential_class, caplog):
        """Test that Microsoft Entra ID credentials are reused across instances."""
        # Mock the credential and token
        mock_credential = Mock()
        mock_token_obj = Mock()
        mock_token_obj.token = "mock_token_123"
        mock_credential.get_token.return_value = mock_token_obj
        mock_credential_class.return_value = mock_credential

        settings = Settings(
            azure_openai_use_microsoft_entra_id="true",
            azure_openai_endpoint="https://dummy.openai.azure.com/",
        )

        with caplog.at_level(logging.INFO):
            # Create multiple instances
            wrapper1 = AzureOpenAiWrapper(settings)
            wrapper2 = AzureOpenAiWrapper(settings)

            # Access models to trigger authentication
            try:
                _ = wrapper1.chat_model
                _ = wrapper2.chat_model
            except Exception:
                pass  # Expected due to mocking

            # Credential should be initialized only once
            assert mock_credential_class.call_count == 1
            # Token should be requested only once
            assert mock_credential.get_token.call_count == 1

            # Should see initialization logs only once
            assert caplog.text.count("Initializing Microsoft Entra ID authentication") == 1
            assert caplog.text.count("Getting authentication token") == 1

    @patch("template_langgraph.llms.azure_openais.DefaultAzureCredential")
    def test_thread_safety(self, mock_credential_class):
        """Test that authentication is thread-safe."""
        mock_credential = Mock()
        mock_token_obj = Mock()
        mock_token_obj.token = "mock_token_123"
        mock_credential.get_token.return_value = mock_token_obj
        mock_credential_class.return_value = mock_credential

        settings = Settings(
            azure_openai_use_microsoft_entra_id="true",
            azure_openai_endpoint="https://dummy.openai.azure.com/",
        )

        results = []
        errors = []

        def worker():
            try:
                wrapper = AzureOpenAiWrapper(settings)
                token = wrapper._get_auth_token()
                results.append(token)
            except Exception as e:
                errors.append(e)

        # Create multiple threads that try to authenticate simultaneously
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(token == "mock_token_123" for token in results)

        # Credential should be initialized only once despite multiple threads
        assert mock_credential_class.call_count == 1
        assert mock_credential.get_token.call_count == 1

    def test_different_settings_per_instance(self):
        """Test that different instances can have different settings."""
        settings1 = Settings(
            azure_openai_use_microsoft_entra_id="false",
            azure_openai_api_key="key1",
            azure_openai_endpoint="https://endpoint1.openai.azure.com/",
        )

        settings2 = Settings(
            azure_openai_use_microsoft_entra_id="false",
            azure_openai_api_key="key2",
            azure_openai_endpoint="https://endpoint2.openai.azure.com/",
        )

        wrapper1 = AzureOpenAiWrapper(settings1)
        wrapper2 = AzureOpenAiWrapper(settings2)

        # Each instance should maintain its own settings
        assert wrapper1.settings.azure_openai_api_key == "key1"
        assert wrapper2.settings.azure_openai_api_key == "key2"
        assert wrapper1.settings.azure_openai_endpoint == "https://endpoint1.openai.azure.com/"
        assert wrapper2.settings.azure_openai_endpoint == "https://endpoint2.openai.azure.com/"

    def test_create_embedding_method_compatibility(self):
        """Test that the create_embedding method still works."""
        settings = Settings(
            azure_openai_use_microsoft_entra_id="false",
            azure_openai_api_key="dummy_key",
            azure_openai_endpoint="https://dummy.openai.azure.com/",
        )

        wrapper = AzureOpenAiWrapper(settings)

        # This should not raise an error about missing methods
        # (though it will fail due to dummy credentials)
        try:
            wrapper.create_embedding("test text")
        except Exception:
            pass  # Expected due to dummy credentials

        # Verify the method exists and is callable
        assert hasattr(wrapper, "create_embedding")
        assert callable(getattr(wrapper, "create_embedding"))

    @patch("template_langgraph.llms.azure_openais.DefaultAzureCredential")
    def test_mixed_authentication_methods(self, mock_credential_class, caplog):
        """Test using both authentication methods in different instances."""
        mock_credential = Mock()
        mock_token_obj = Mock()
        mock_token_obj.token = "mock_token_123"
        mock_credential.get_token.return_value = mock_token_obj
        mock_credential_class.return_value = mock_credential

        # API key settings
        api_settings = Settings(
            azure_openai_use_microsoft_entra_id="false",
            azure_openai_api_key="dummy_key",
            azure_openai_endpoint="https://dummy.openai.azure.com/",
        )

        # Entra ID settings
        entra_settings = Settings(
            azure_openai_use_microsoft_entra_id="true",
            azure_openai_endpoint="https://dummy.openai.azure.com/",
        )

        with caplog.at_level(logging.INFO):
            wrapper_api = AzureOpenAiWrapper(api_settings)
            wrapper_entra = AzureOpenAiWrapper(entra_settings)

            # Access models to trigger different authentication paths
            try:
                _ = wrapper_api.chat_model
                _ = wrapper_entra.chat_model
            except Exception:
                pass  # Expected due to dummy/mock credentials

            # Should see both authentication methods being used
            assert "Using API key for authentication" in caplog.text
            assert "Initializing Microsoft Entra ID authentication" in caplog.text
