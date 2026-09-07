import pytest
from promptcraft.core.providers import BaseProvider
from typing import Optional

class MockLLMProvider(BaseProvider):
    """Provedor falso que retorna uma resposta simulada sem chamar APIs externas."""
    def __init__(self, response_to_return: str = "{}"):
        self.response_to_return = response_to_return
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self.response_to_return

@pytest.fixture
def mock_provider():
    return MockLLMProvider()