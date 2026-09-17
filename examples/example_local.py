# test_local.py
from promptcraft.finance.invoice_parser import InvoiceParser

# O GeminiProvider já sobe pegando a chave direto do .env
parser = InvoiceParser()

texto_nota = """
AUTO POSTO EXEMPLO LTDA
CNPJ: 12.345.678/0001-95
DATA: 07/09/2026 10:15:00
VALOR TOTAL R$ 220,50
CHAVE DE ACESSO: 3526 0912 3456 7800 0195 5500 1000 0543 2110 1234 5678
"""

resultado = parser.parse(texto_nota)
print(resultado.model_dump_json(indent=2))