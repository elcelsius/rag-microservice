import pytest
from langchain.schema import Document

import query_handler


def test_build_metadata_answer_returns_contacts():
    docs = [
        Document(
            page_content="Contato da DTI",
            metadata={
                "emails": ["dti.fc@unesp.br"],
                "phones": ["(14) 3103-6000"],
            },
        ),
        Document(
            page_content="Outro contato",
            metadata={
                "emails": ["suporte.fc@unesp.br"],
                "phones": ["(14) 3103-6001"],
            },
        ),
    ]
    dbg = {}

    answer = query_handler._build_metadata_answer(
        "Qual o e-mail e telefone da secretaria da DTI?",
        docs,
        dbg,
    )

    assert answer is not None
    assert "dti.fc@unesp.br" in answer
    assert "(14) 3103-6000" in answer
    direct = dbg["metadata"]["direct_answer"]
    assert direct["emails"][0] == "dti.fc@unesp.br"
    assert "(14) 3103-6000" in direct["phones"]


def test_finalize_result_short_circuits_to_metadata(monkeypatch, mocker):
    monkeypatch.setattr(query_handler, "STRUCTURED_ANSWER", False)
    llm_mock = mocker.patch("query_handler._llm_resposta_final")

    docs = [
        Document(
            page_content="Secretaria DTI atendimento",
            metadata={
                "emails": ["dti.fc@unesp.br"],
                "phones": ["(14) 3103-6000"],
                "source": "fc/dti_contato.txt",
                "chunk": 1,
            },
        )
    ]

    dbg = {"timing_ms": {}}

    result = query_handler._finalize_result(
        "Qual o e-mail da secretaria da DTI?",
        docs,
        0.82,
        dbg,
    )

    assert result["answer"].startswith("### Contato")
    assert "dti.fc@unesp.br" in result["answer"]
    assert result["citations"][0]["source"] == "fc/dti_contato.txt"
    assert result["confidence"] == 0.82
    llm_mock.assert_not_called()


def test_format_contact_answer_limits_results():
    entry = {
        "name": "Secretaria DTI",
        "phones": ["(14) 3103-6008", "(14) 3103-6023", "(14) 3103-9999"],
        "emails": [
            "dti.fc@unesp.br",
            "suporte.fc@unesp.br",
            "agnelo.rodrigues@unesp.br",
            "outra.pessoa@unesp.br",
        ],
        "departments": {"Diretoria Técnica de Informática"},
    }

    answer = query_handler._format_contact_answer(entry, "qual é o e-mail da secretaria da dti?")

    assert answer.count("@unesp.br") <= 3
    assert answer.count("(14)") <= 2
