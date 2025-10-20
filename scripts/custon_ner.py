#!/usr/bin/env python3
# scripts/custom_ner.py
from pathlib import Path
from typing import Dict, Iterable, List

import spacy
import yaml
from spacy.language import Language
from spacy.pipeline import EntityRuler


def _load_terms(path: Path) -> Dict[str, Dict[str, str | Iterable[str]]]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("terms.yml inválido: root não é um objeto YAML")
    return data


def _to_patterns(section: Dict[str, str | Iterable[str]], label: str, nlp: Language) -> List[Dict]:
    """
    Gera padrões de token case-insensitive para o EntityRuler 
    usando o tokenizador do nlp.
    """
    patterns: List[Dict] = []
    if not section:
        return patterns

    for slug, value in section.items():
        # --- INÍCIO DA CORREÇÃO ---
        # Cria um conjunto (set) para guardar todos os textos (evita duplicatas)
        all_texts_to_match = set()

        # 1. Adiciona a própria chave (slug) como um padrão
        all_texts_to_match.add(slug)

        # 2. Adiciona os valores da lista (se for uma lista)
        if isinstance(value, list):
            all_texts_to_match.update(val for val in value if val)
        # 3. Adiciona o valor (se for uma string única)
        elif isinstance(value, str):
            all_texts_to_match.add(value)
        # --- FIM DA CORREÇÃO ---

        for text in all_texts_to_match:
            doc = nlp.make_doc(text)
            pattern_tokens = [{"LOWER": token.text.lower()} for token in doc if not token.is_space]
            
            if pattern_tokens:
                patterns.append({"label": label, "pattern": pattern_tokens, "id": slug})
                
    return patterns


def create_custom_ner_pipeline(terms_path: str) -> Language:
    """
    Cria um pipeline spaCy com um EntityRuler customizado 
    a partir de um arquivo de termos YAML.
    """
    # 1. Carrega o modelo base
    nlp = spacy.load("pt_core_news_lg")
    
    # 2. Carrega os termos do arquivo YAML
    terms = _load_terms(Path(terms_path))

    # 3. Adiciona a fábrica "entity_ruler" ao pipeline
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.overwrite_ents = True

    # 4. Carrega os padrões passando o 'nlp' para o _to_patterns 
    #    para que ele possa usar o tokenizador
    dept_patterns = _to_patterns(terms.get("departments", {}), "DEPARTAMENTO", nlp)
    syn_patterns = _to_patterns(terms.get("synonyms", {}), "SINONIMO", nlp)
    
    # --- CORREÇÃO PRINCIPAL ---
    # 5. Adiciona o carregamento dos aliases como PESSOA
    alias_patterns = _to_patterns(terms.get("aliases", {}), "PESSOA", nlp)
    
    # 6. Adiciona TODOS os padrões ao ruler
    ruler.add_patterns(dept_patterns + syn_patterns + alias_patterns)
    # --- FIM DA CORREÇÃO ---

    # 7. Retorna o pipeline modificado
    return nlp


if __name__ == "__main__":
    import sys  # Importe o 'sys' no topo do arquivo ou aqui

    # Pega os argumentos da linha de comando, exceto o nome do script
    args = sys.argv[1:]
    
    # Se nenhum argumento foi passado, mostra um erro
    if not args:
        print("Erro: Forneça uma frase para analisar.")
        print("Exemplo: python3 scripts/custon_ner.py qual o telefone da andreia")
        sys.exit(1)

    # Junta todos os argumentos em uma única frase
    text_to_analyze = " ".join(args)
    
    print(f"Analisando texto: '{text_to_analyze}'")
    print("---")
    
    # Cria o pipeline
    nlp = create_custom_ner_pipeline("config/ontology/terms.yml")
    
    # Processa o texto fornecido
    doc = nlp(text_to_analyze)
    
    # Imprime as entidades encontradas
    for ent in doc.ents:
        print(f"Texto: {ent.text:<25} Rótulo: {ent.label_:<15} ID: {ent.kb_id_}")
