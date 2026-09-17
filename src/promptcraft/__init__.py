"""
PromptCraft AI - Prompts de IA tipados e prontos para casos de uso de negócio no Brasil.
Suporta múltiplos provedores de LLM: Gemini, OpenAI, Claude, Ollama, Groq, DeepSeek, etc.
"""

from promptcraft.core.providers import (
    BaseProvider,
    GeminiProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    get_provider,
)
from promptcraft.core.utils import clean_json_output
from promptcraft.finance.invoice_parser import InvoiceParser
from promptcraft.finance.models import InvoiceData

__version__ = "0.2.0"

__all__ = [
    "BaseProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "get_provider",
    "clean_json_output",
    "InvoiceParser",
    "InvoiceData",
]
