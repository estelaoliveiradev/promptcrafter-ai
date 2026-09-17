# PromptCraft AI 🚀

[![PyPI Version](https://img.shields.io/pypi/v/promptcraft-ai.svg?color=blue)](https://pypi.org/project/promptcraft-ai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/promptcraft-ai.svg)](https://pypi.org/project/promptcraft-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Prompts de IA tipados e prontos para casos de uso de negócio no Brasil, totalmente **agnóstico a LLMs**. Utilize **Google Gemini**, **OpenAI**, **Anthropic Claude**, **Ollama (modelos locais)** ou qualquer API compatível com OpenAI (**Groq**, **DeepSeek**, **OpenRouter**) com a mesma interface simples e tipada com Pydantic.

---

## 📦 Instalação

### Instalação Básica (com suporte a Gemini e Ollama):
```bash
pip install promptcraft-ai
```

### Instalação com suporte a outros provedores:
```bash
# Com suporte a OpenAI / Groq / DeepSeek
pip install "promptcraft-ai[openai]"

# Com suporte a Anthropic Claude
pip install "promptcraft-ai[anthropic]"

# Pacote completo com todos os provedores
pip install "promptcraft-ai[all]"
```

---

## ⚡ Como Usar

### 1. Detecção Automática (Zero-Config)
Defina a chave no seu arquivo `.env` (ex: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` ou `GROQ_API_KEY`). O PromptCraft detecta automaticamente:

```python
from promptcraft import InvoiceParser

# Auto-detecta o provedor disponível no ambiente
parser = InvoiceParser()

texto_ocr = """
AUTO POSTO EXEMPLO LTDA
CNPJ: 12.345.678/0001-95
DATA: 07/09/2026 10:15:00
VALOR TOTAL R$ 220,50
CHAVE DE ACESSO: 3526 0912 3456 7800 0195 5500 1000 0543 2110 1234 5678
"""

dados = parser.parse(texto_ocr)
print(dados.empresa_emitente)  # "AUTO POSTO EXEMPLO LTDA"
print(dados.cnpj_emitente)      # "12345678000195"
print(dados.valor_total)        # 220.50
```

---

### 2. Escolhendo a LLM por Nome (String)

Você pode selecionar o provedor diretamente como string:

```python
from promptcraft import InvoiceParser

# Usando Google Gemini
parser = InvoiceParser(provider="gemini")

# Usando OpenAI
parser = InvoiceParser(provider="openai", model="gpt-4o-mini")

# Usando Anthropic Claude
parser = InvoiceParser(provider="anthropic", model="claude-3-5-haiku-latest")

# Usando Ollama (100% Local e Grátis, sem necessidade de API Key)
parser = InvoiceParser(provider="ollama", model="llama3")

# Usando Groq (Ultra rápido)
parser = InvoiceParser(provider="groq", model="llama-3.3-70b-versatile")

# Usando DeepSeek
parser = InvoiceParser(provider="deepseek", model="deepseek-chat")
```

---

### 3. Instanciando Provedores Explicitamente

Você também pode importar e configurar as classes de provedores com parâmetros customizados:

```python
from promptcraft import (
    InvoiceParser,
    GeminiProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)

# OpenAI com base_url customizada (ex: LM Studio, vLLM ou proxy corporativo)
provider_local = OpenAIProvider(
    base_url="http://localhost:1234/v1",
    api_key="nao-obrigatoria",
    model="local-model"
)
parser = InvoiceParser(provider=provider_local)

# Ollama em servidor remoto na rede local
provider_ollama = OllamaProvider(
    host="http://192.168.1.100:11434",
    model="qwen2.5:7b"
)
parser = InvoiceParser(provider=provider_ollama)
```

---

### 4. Criando seu Próprio Provedor Customizado

Se você usa uma LLM interna ou outro framework, basta herdar de `BaseProvider`:

```python
from promptcraft import BaseProvider, InvoiceParser
from typing import Optional

class MeuProvedorProprio(BaseProvider):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Sua lógica de chamada aqui
        return '{"empresa_emitente": "Empresa X", "valor_total": 100.0}'

parser = InvoiceParser(provider=MeuProvedorProprio())
```

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.