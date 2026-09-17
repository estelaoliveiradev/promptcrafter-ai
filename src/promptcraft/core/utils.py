import re

def clean_json_output(raw_text: str) -> str:
    """
    Limpa e extrai a cadeia JSON de uma saída bruta de LLM,
    removendo marcações Markdown (```json ... ```) e textos explicativos adicionais.
    """
    text = raw_text.strip()

    # Padrão 1: Bloco de código Markdown ```json ... ``` ou ``` ... ```
    match_codeblock = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match_codeblock:
        text = match_codeblock.group(1).strip()

    # Padrão 2: Localiza o primeiro '{' e o último '}' para extrair o objeto JSON
    primeira_chave = text.find("{")
    ultima_chave = text.rfind("}")
    if primeira_chave != -1 and ultima_chave != -1 and ultima_chave > primeira_chave:
        return text[primeira_chave : ultima_chave + 1]

    # Padrão 3: Caso seja uma lista JSON [...]
    primeiro_colchete = text.find("[")
    ultimo_colchete = text.rfind("]")
    if primeiro_colchete != -1 and ultimo_colchete != -1 and ultimo_colchete > primeiro_colchete:
        return text[primeiro_colchete : ultimo_colchete + 1]

    return text
