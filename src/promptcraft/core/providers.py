import os
import time
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Dict
from dotenv import load_dotenv, find_dotenv

# Carrega variáveis do arquivo .env (recursivamente na pasta atual e diretórios superiores)
load_dotenv(find_dotenv(usecwd=True))


class BaseProvider(ABC):
    """Classe base abstrata para todos os provedores de LLM no PromptCraft."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Gera resposta da LLM a partir de um prompt e um system prompt opcional."""
        pass


class GeminiProvider(BaseProvider):
    """Provedor para os modelos do Google Gemini via SDK google-genai."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-flash-latest"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            caminho_local_env = os.path.join(os.getcwd(), ".env")
            if os.path.exists(caminho_local_env):
                load_dotenv(caminho_local_env)
                self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. Certifique-se de definir a variável de ambiente, "
                "ter um arquivo .env no projeto ou passar api_key='...' explicitamente."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "O pacote 'google-genai' é necessário para usar o GeminiProvider. "
                "Instale-o com: pip install google-genai"
            ) from exc

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            system_instruction=system_prompt if system_prompt else None,
        )

        max_tentativas = 3
        espera = 2  # segundos

        for tentativa in range(1, max_tentativas + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text

            except Exception as e:
                eh_503 = "503" in str(e) or "overloaded" in str(e).lower()

                if eh_503 and tentativa < max_tentativas:
                    print(f"⚠️ Servidor do Gemini ocupado (503). Tentando novamente em {espera}s (tentativa {tentativa}/{max_tentativas})...")
                    time.sleep(espera)
                    espera *= 2
                else:
                    raise e


class OpenAIProvider(BaseProvider):
    """
    Provedor para OpenAI e qualquer endpoint compatível
    (Groq, DeepSeek, OpenRouter, Mistral, LM Studio, vLLM, etc.).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self.temperature = temperature
        self.extra_headers = extra_headers

        if not self.api_key and not self.base_url:
            caminho_local_env = os.path.join(os.getcwd(), ".env")
            if os.path.exists(caminho_local_env):
                load_dotenv(caminho_local_env)
                self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key and not self.base_url:
            raise ValueError(
                "OPENAI_API_KEY não encontrada. Defina a variável de ambiente OPENAI_API_KEY "
                "ou passe api_key='...' explicitamente."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "O pacote 'openai' é necessário para usar o OpenAIProvider. "
                "Instale-o com: pip install openai"
            ) from exc

        # Se base_url foi passada (ex: Ollama, Groq, LM Studio) mas api_key está ausente, usa placeholder
        client_key = self.api_key or "no-key-required"
        self.client = OpenAI(
            api_key=client_key,
            base_url=self.base_url,
            default_headers=self.extra_headers,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Tenta modo json_object para saídas estruturadas se suportado
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback sem response_format estrito para modelos compatíveis que não suportam json_object
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

        content = response.choices[0].message.content
        return content or ""


class AnthropicProvider(BaseProvider):
    """Provedor para os modelos Claude da Anthropic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-haiku-latest",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            caminho_local_env = os.path.join(os.getcwd(), ".env")
            if os.path.exists(caminho_local_env):
                load_dotenv(caminho_local_env)
                self.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY não encontrada. Defina a variável de ambiente ANTHROPIC_API_KEY "
                "ou passe api_key='...' explicitamente."
            )

        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "O pacote 'anthropic' é necessário para usar o AnthropicProvider. "
                "Instale-o com: pip install anthropic"
            ) from exc

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        # Anthropic retorna blocos de conteúdo
        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        return "".join(text_parts)


class OllamaProvider(BaseProvider):
    """
    Provedor para modelos executados localmente via Ollama (Llama 3, Mistral, Qwen, etc.).
    Não requer chaves de API nem pacotes externos (usa urllib padrão).
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        data = json.dumps(payload).encode("utf-8")
        url = f"{self.host}/api/chat"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Não foi possível conectar ao Ollama em '{self.host}'. "
                f"Certifique-se de que o Ollama está instalado e em execução ('ollama serve'). Detalhes: {e}"
            ) from e


def get_provider(
    provider: Optional[Union[str, BaseProvider]] = None,
    **kwargs: Any,
) -> BaseProvider:
    """
    Fábrica inteligente de provedores de LLM.

    Permite obter um provedor por instância, por nome ('gemini', 'openai', 'anthropic', 'ollama', 'groq', 'deepseek')
    ou automaticamente detectando chaves de API disponíveis no ambiente (.env).
    """
    if isinstance(provider, BaseProvider):
        return provider

    if isinstance(provider, str):
        provider_lower = provider.strip().lower()
        if provider_lower in ("gemini", "google"):
            return GeminiProvider(**kwargs)
        elif provider_lower in ("openai", "gpt"):
            return OpenAIProvider(**kwargs)
        elif provider_lower in ("anthropic", "claude"):
            return AnthropicProvider(**kwargs)
        elif provider_lower == "ollama":
            return OllamaProvider(**kwargs)
        elif provider_lower == "groq":
            groq_key = kwargs.pop("api_key", None) or os.getenv("GROQ_API_KEY")
            groq_model = kwargs.pop("model", "llama-3.3-70b-versatile")
            return OpenAIProvider(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
                model=groq_model,
                **kwargs,
            )
        elif provider_lower == "deepseek":
            deepseek_key = kwargs.pop("api_key", None) or os.getenv("DEEPSEEK_API_KEY")
            deepseek_model = kwargs.pop("model", "deepseek-chat")
            return OpenAIProvider(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com",
                model=deepseek_model,
                **kwargs,
            )
        elif provider_lower == "openrouter":
            openrouter_key = kwargs.pop("api_key", None) or os.getenv("OPENROUTER_API_KEY")
            return OpenAIProvider(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                **kwargs,
            )
        else:
            raise ValueError(
                f"Provedor desconhecido: '{provider}'. "
                f"Opções disponíveis: 'gemini', 'openai', 'anthropic', 'ollama', 'groq', 'deepseek', 'openrouter' "
                f"ou instancie sua própria classe herdando de BaseProvider."
            )

    # Detecção automática se nenhum provedor for informado
    if os.getenv("GEMINI_API_KEY"):
        return GeminiProvider(**kwargs)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider(**kwargs)
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicProvider(**kwargs)
    if os.getenv("GROQ_API_KEY"):
        return OpenAIProvider(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model=kwargs.pop("model", "llama-3.3-70b-versatile"),
            **kwargs,
        )

    # Se nada foi encontrado, tenta fallback padrão do Gemini (que exibirá mensagem de chave ausente)
    # ou levanta erro informativo
    raise ValueError(
        "Nenhum provedor de LLM configurado. "
        "Defina uma variável de ambiente (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY) "
        "ou especifique o provedor explicitamente. Exemplo:\n"
        "  parser = InvoiceParser(provider='ollama')\n"
        "  parser = InvoiceParser(provider=OpenAIProvider(api_key='...'))"
    )