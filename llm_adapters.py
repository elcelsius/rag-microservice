# llm_adapters.py
# Este módulo contém classes adaptadoras para fazer com que diferentes LLMs
# ou SDKs se comportem de maneira consistente com as interfaces esperadas por
# bibliotecas como LangChain ou RAGAs.

import os
import asyncio
from typing import Any, List, Optional, Dict

import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import root_validator


class RagasGoogleApiLLM(LLM):
    """Adaptador para usar o SDK do Google GenAI diretamente com o RAGAs.
    
    Esta classe herda de langchain_core.language_models.llms.LLM e implementa
    a interface mínima necessária para ser compatível com o RAGAs, evitando
    conflitos de concorrência que podem ocorrer com o wrapper padrão do LangChain.
    """
    model_name: str
    model: Any = None

    @root_validator(pre=False, skip_on_failure=True)
    def _initialize_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Inicializa o cliente do modelo Google GenAI após a validação do Pydantic."""
        if "model_name" in values:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("A chave da API do Google (GOOGLE_API_KEY ou GEMINI_API_KEY) não foi encontrada.")
            genai.configure(api_key=api_key)
            values["model"] = genai.GenerativeModel(values["model_name"])
        return values

    @property
    def _llm_type(self) -> str:
        return "ragas_google_api_llm"

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        """Método de chamada síncrona obrigatório da classe LLM."""
        prompt_text = str(prompt)
        if self.model is None:
            raise ValueError("Modelo GenAI não inicializado. Verifique a configuração.")
            
        try:
            # Extrai a temperatura dos kwargs ou usa 0.0 como padrão
            temperature = kwargs.get("temperature", 0.0)
            response = self.model.generate_content(prompt_text, generation_config={"temperature": temperature})
            return response.text
        except Exception as e:
            print(f"Erro na chamada da API do Google: {e}")
            return ""

    def set_run_config(self, run_config: Any):
        """Método obrigatório para compatibilidade com o executor do RAGAs."""
        pass
