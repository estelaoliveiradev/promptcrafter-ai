import json
import pytest
from unittest.mock import MagicMock, patch
import urllib.error

from promptcraft.core.utils import clean_json_output
from promptcraft.core.providers import (
    BaseProvider,
    GeminiProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    get_provider,
)
from promptcraft.finance.invoice_parser import InvoiceParser
from tests.conftest import MockLLMProvider


class TestCleanJsonOutput:
    def test_pure_json(self):
        raw = '{"name": "test", "value": 123}'
        assert clean_json_output(raw) == '{"name": "test", "value": 123}'

    def test_markdown_code_block(self):
        raw = """```json
        {"name": "test", "value": 123}
        ```"""
        assert json.loads(clean_json_output(raw)) == {"name": "test", "value": 123}

    def test_markdown_without_json_tag(self):
        raw = """```
        {"chave": "valor"}
        ```"""
        assert json.loads(clean_json_output(raw)) == {"chave": "valor"}

    def test_conversational_text_around_json(self):
        raw = """Aqui está a sua resposta:
        {"empresa": "ABC", "total": 99.9}
        Espero ter ajudado!"""
        assert json.loads(clean_json_output(raw)) == {"empresa": "ABC", "total": 99.9}

    def test_json_array(self):
        raw = "Resultados: [1, 2, 3]"
        assert clean_json_output(raw) == "[1, 2, 3]"


class TestGetProvider:
    def test_returns_instance_if_already_provider(self):
        mock = MockLLMProvider()
        assert get_provider(mock) is mock

    def test_get_ollama_provider(self):
        provider = get_provider("ollama", model="mistral", host="http://localhost:11434")
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "mistral"
        assert provider.host == "http://localhost:11434"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError) as exc:
            get_provider("invalid_provider_name")
        assert "Provedor desconhecido" in str(exc.value)

    def test_autodetect_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with patch("google.genai.Client"):
            provider = get_provider()
            assert isinstance(provider, GeminiProvider)

    def test_autodetect_openai(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            provider = get_provider()
            assert isinstance(provider, OpenAIProvider)

    def test_openai_missing_package_raises_import_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError) as exc:
                OpenAIProvider()
            assert "pip install openai" in str(exc.value)

    def test_anthropic_missing_package_raises_import_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError) as exc:
                AnthropicProvider()
            assert "pip install anthropic" in str(exc.value)

    def test_autodetect_none_configured_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc:
            get_provider()
        assert "Nenhum provedor de LLM configurado" in str(exc.value)


class TestOllamaProvider:
    def test_ollama_generate_success(self):
        provider = OllamaProvider(model="llama3")

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "message": {"content": '{"ok": true}'}
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            resultado = provider.generate("teste prompt", system_prompt="sistema")
            assert resultado == '{"ok": true}'
            assert mock_urlopen.called

    def test_ollama_connection_error(self):
        provider = OllamaProvider(model="llama3", host="http://localhost:99999")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(ConnectionError) as exc:
                provider.generate("teste")
            assert "Não foi possível conectar ao Ollama" in str(exc.value)


class TestInvoiceParserWithProviders:
    def test_invoice_parser_accepts_provider_instance(self):
        mock = MockLLMProvider(response_to_return='{"empresa_emitente": "TESTE", "valor_total": 10.0}')
        parser = InvoiceParser(provider=mock)
        res = parser.parse("nota fiscal texto")
        assert res.empresa_emitente == "TESTE"
        assert res.valor_total == 10.0

    def test_invoice_parser_accepts_provider_string(self):
        parser = InvoiceParser(provider="ollama", model="llama3")
        assert isinstance(parser.provider, OllamaProvider)
        assert parser.provider.model == "llama3"
