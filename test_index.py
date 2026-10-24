"""
Проверка индекса: по вопросу находится чанк из нужного документа.

Запуск: uv run pytest test_index.py
"""

from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS

from build_index import load_index

CASES = [
    ("На какой планете родился Kael Solvane?", "kael_solvane.md"),
    ("Что такое Void Core и чем он вооружён?", "vc_1_orbital_battle_station.md"),
    ("Кто такие Grakkar и откуда они родом?", "grakkar.md"),
]


@pytest.fixture(scope="module")
def index() -> FAISS:
    return load_index()


@pytest.mark.parametrize(("query", "expected"), CASES)
def test_query_finds_document(index: FAISS, query: str, expected: str) -> None:
    found = [Path(doc.metadata["source"]).name for doc in index.similarity_search(query, k=3)]
    assert expected in found
