import pytest
from src.promptcraft.finance.invoice_parser import InvoiceParser
from src.promptcraft.finance.models import InvoiceData
from tests.conftest import MockLLMProvider

def test_invoice_parser_sucesso_completo():
    # 1. Simula retorno válido da LLM
    json_simulado = """
    {
        "empresa_emitente": "AUTO POSTO EXEMPLO LTDA",
        "cnpj_emitente": "12345678000195",
        "data_emissao": "07/09/2026",
        "valor_total": 220.50,
        "chave_acesso": "35260912345678000195550010000543211012345678"
    }
    """
    provider = MockLLMProvider(response_to_return=json_simulado)
    parser = InvoiceParser(provider=provider)

    texto_ocr = "AUTO POSTO EXEMPLO LTDA CNPJ: 12.345.678/0001-95 VALOR R$ 220,50"
    resultado = parser.parse(texto_ocr)

    # 2. Asserções
    assert isinstance(resultado, InvoiceData)
    assert resultado.empresa_emitente == "AUTO POSTO EXEMPLO LTDA"
    assert resultado.cnpj_emitente == "12345678000195"
    assert resultado.data_emissao == "07/09/2026"
    assert resultado.valor_total == 220.50
    assert resultado.chave_acesso == "35260912345678000195550010000543211012345678"


def test_invoice_parser_campos_ausentes():
    # Simula cupom fiscal simples que não possui chave de acesso nem CNPJ
    json_simulado = """
    {
        "empresa_emitente": "PADARIA CENTRAL",
        "cnpj_emitente": null,
        "data_emissao": "01/01/2026",
        "valor_total": 15.00,
        "chave_acesso": null
    }
    """
    provider = MockLLMProvider(response_to_return=json_simulado)
    parser = InvoiceParser(provider=provider)

    resultado = parser.parse("PADARIA CENTRAL VALOR R$ 15,00")

    assert resultado.empresa_emitente == "PADARIA CENTRAL"
    assert resultado.cnpj_emitente is None
    assert resultado.chave_acesso is None
    assert resultado.valor_total == 15.00


def test_invoice_parser_trata_markdown_tags():
    # Testa se o parser remove as crases ```json ... ``` caso a LLM insira
    json_com_markdown = """```json
    {
        "empresa_emitente": "LOJA ABC",
        "cnpj_emitente": "98765432000100",
        "data_emissao": null,
        "valor_total": 50.0,
        "chave_acesso": null
    }
    ```"""
    provider = MockLLMProvider(response_to_return=json_com_markdown)
    parser = InvoiceParser(provider=provider)

    resultado = parser.parse("LOJA ABC TOTAL 50.0")

    assert resultado.empresa_emitente == "LOJA ABC"
    assert resultado.valor_total == 50.0


def test_invoice_parser_lanca_erro_em_json_invalido():
    # Simula uma resposta truncada ou alucinação que não forma um JSON válido
    resposta_quebrada = "Desculpe, não encontrei os dados na nota fiscal."
    provider = MockLLMProvider(response_to_return=resposta_quebrada)
    parser = InvoiceParser(provider=provider)

    with pytest.raises(ValueError) as exc_info:
        parser.parse("texto qualquer")

    assert "Falha ao validar resposta" in str(exc_info.value)