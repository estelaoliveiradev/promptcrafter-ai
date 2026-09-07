import os
import time
from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Procura o arquivo .env recursivamente na pasta atual e diretórios superiores
load_dotenv(find_dotenv(usecwd=True))

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass


class GeminiProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-flash-latest"):
        # Se ainda não encontrou, tenta buscar especificamente no diretório atual de execução
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            # Fallback explícito para tentar achar no diretório de execução atual
            caminho_local_env = os.path.join(os.getcwd(), ".env")
            if os.path.exists(caminho_local_env):
                load_dotenv(caminho_local_env)
                self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. Certifique-se de definir a variável de ambiente, "
                "ter um arquivo .env no projeto ou passar api_key='...' explicitamente."
            )
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            system_instruction=system_prompt if system_prompt else None,
        )

        # Política de retentativa para contornar o 503
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