# llm_adapters.py
# Este módulo contém classes adaptadoras para fazer com que diferentes LLMs
# ou SDKs se comportem de maneira consistente com as interfaces esperadas por
# bibliotecas como LangChain ou RAGAs.

import os
import asyncio
from typing import Any, List, Optional, Dict

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

class RagasGoogleApiLLM(LLM):
    """Adaptador para usar o SDK do Google GenAI diretamente com o RAGAs.
    
    Esta classe herda de langchain_core.language_models.llms.LLM e implementa
    a interface mínima necessária para ser compatível com o RAGAs, evitando
    conflitos de concorrência que podem ocorrer com o wrapper padrão do LangChain.
    """
    model_name: str
    model: Any = None
    google_api_key: Optional[str] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.model_name:
            raise ValueError("O argumento 'model_name' é obrigatório.")

        self.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.google_api_key:
            raise ValueError("A chave da API do Google (GOOGLE_API_KEY ou GEMINI_API_KEY) não foi encontrada.")
        
        try:
            self.model = GoogleGenerativeAI(model=self.model_name, google_api_key=self.google_api_key)
        except Exception as e:
            raise ValueError(f"Falha ao inicializar o modelo GoogleGenerativeAI: {e}") from e

    @property
    def _llm_type(self) -> str:
        return "ragas_google_api_llm"

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """Método de chamada síncrona obrigatório da classe LLM."""
        prompt_text = str(prompt)
        if self.model is None:
            # Esta verificação é agora um pouco redundante, mas boa para segurança
            raise ValueError("Modelo GenAI não inicializado. Verifique a configuração.")
            
        try:
            # Extrai a temperatura dos kwargs ou usa 0.0 como padrão
            temperature = kwargs.get("temperature", 0.0)
            response = self.model.invoke(prompt_text, temperature=temperature)
            return response
        except Exception as e:
            print(f"Erro na chamada da API do Google: {e}")
            return ""

    def set_run_config(self, run_config: Any):
        """Método obrigatório para compatibilidade com o executor do RAGAs."""
        pass
