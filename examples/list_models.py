import os
from dotenv import load_dotenv
from google import genai

# Carrega as variáveis do .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY não encontrada no arquivo .env")
    exit(1)

# Inicializa o cliente do SDK novo
client = genai.Client(api_key=api_key)

print("🔍 Consultando modelos disponíveis para a sua chave...\n")

try:
    modelos = client.models.list()
    
    modelos_encontrados = 0
    for model in modelos:
        # Filtra apenas os modelos que suportam geração de conteúdo
        metodos = getattr(model, "supported_generation_methods", []) or []
        
        # Exibe o modelo se ele suportar generateContent ou se for da família gemini
        if "generateContent" in metodos or "gemini" in model.name.lower():
            modelos_encontrados += 1
            print(f"📌 {model.name}")
            if hasattr(model, "display_name") and model.display_name:
                print(f"   Nome de Exibição: {model.display_name}")
            if hasattr(model, "description") and model.description:
                print(f"   Descrição: {model.description[:80]}...")
            print("-" * 50)

    if modelos_encontrados == 0:
        print("Nenhum modelo compatível encontrado. Verifique as permissões da chave.")

except Exception as e:
    print(f"❌ Erro ao listar modelos: {e}")