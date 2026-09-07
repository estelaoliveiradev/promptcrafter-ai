import os
import time
from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass


class GeminiProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-flash-latest"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY não encontrada no .env")
        
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
                # Verifica se o erro é 503 ou sobrecarga de servidor
                eh_503 = "503" in str(e) or "overloaded" in str(e).lower()
                
                if eh_503 and tentativa < max_tentativas:
                    print(f"⚠️ Servidor do Gemini ocupado (503). Tentando novamente em {espera}s (tentativa {tentativa}/{max_tentativas})...")
                    time.sleep(espera)
                    espera *= 2  # dobra o tempo de espera (2s -> 4s)
                else:
                    raise e