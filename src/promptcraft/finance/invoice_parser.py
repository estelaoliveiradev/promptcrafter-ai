import json
from typing import Optional, Union, Any
from promptcraft.core.providers import BaseProvider, get_provider
from promptcraft.core.utils import clean_json_output
from promptcraft.finance.models import InvoiceData


class InvoiceParser:
    """
    Parser especializado na extração de dados fiscais e contábeis brasileiros
    (Nota Fiscal, Cupom Fiscal, Danfe, NFC-e) a partir de texto extraído via OCR.
    Compatível com múltiplos provedores de LLM (Gemini, OpenAI, Claude, Ollama, etc.).
    """

    def __init__(
        self,
        provider: Optional[Union[BaseProvider, str]] = None,
        **provider_kwargs: Any,
    ):
        """
        Inicializa o InvoiceParser.

        :param provider: Instância de BaseProvider, nome do provedor ('gemini', 'openai', 'anthropic', 'ollama', etc.),
                         ou None para detecção automática de variáveis de ambiente no .env.
        :param provider_kwargs: Argumentos adicionais repassados para a inicialização do provedor (ex: model, api_key).
        """
        self.provider: BaseProvider = get_provider(provider, **provider_kwargs)

    def parse(self, ocr_text: str) -> InvoiceData:
        system_prompt = (
            "Você é um analisador de documentos fiscais e contábeis brasileiros de alta precisão. "
            "Sua responsabilidade é extrair informações estritamente a partir do texto fornecido."
        )

        user_prompt = f"""
Analise o texto bruto extraído de uma nota fiscal ou cupom fiscal brasileiro via OCR.
Extraia as informações solicitadas no formato JSON exato.

Texto da Nota:
---
{ocr_text}
---

Campos a extrair:
- empresa_emitente: Razão Social ou Nome Fantasia (string ou null).
- cnpj_emitente: Apenas os 14 números do CNPJ, sem pontuações (string ou null).
- data_emissao: Formato DD/MM/AAAA (string ou null).
- valor_total: Número decimal com ponto (float ou null, ex: 123.45).
- chave_acesso: 44 números sequenciais se constar no texto (string ou null).
"""
        raw_output = self.provider.generate(prompt=user_prompt, system_prompt=system_prompt)

        try:
            cleaned_output = clean_json_output(raw_output)
            data_dict = json.loads(cleaned_output)
            return InvoiceData(**data_dict)
        except (json.JSONDecodeError, ValueError) as err:
            raise ValueError(f"Falha ao validar resposta da LLM. Saída bruta: {raw_output}") from err