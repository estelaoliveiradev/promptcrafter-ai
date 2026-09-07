import json
from typing import Optional
from promptcraft.core.providers import BaseProvider, GeminiProvider
from promptcraft.finance.models import InvoiceData

class InvoiceParser:
    def __init__(self, provider: Optional[BaseProvider] = None):
        # Se o usuário não passar nada, assume o Gemini por padrão
        self.provider = provider or GeminiProvider()

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
            data_dict = json.loads(raw_output)
            return InvoiceData(**data_dict)
        except (json.JSONDecodeError, ValueError) as err:
            raise ValueError(f"Falha ao validar resposta do Gemini. Saída bruta: {raw_output}") from err