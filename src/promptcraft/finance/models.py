from pydantic import BaseModel, Field
from typing import Optional

class InvoiceData(BaseModel):
    empresa_emitente: Optional[str] = Field(None, description="Razão Social ou Nome Fantasia do emissor")
    cnpj_emitente: Optional[str] = Field(None, description="CNPJ do emissor (apenas dígitos)")
    data_emissao: Optional[str] = Field(None, description="Data da emissão no formato DD/MM/AAAA")
    valor_total: Optional[float] = Field(None, description="Valor total da nota fiscal em formato decimal")
    chave_acesso: Optional[str] = Field(None, description="Chave de acesso de 44 dígitos, se presente")