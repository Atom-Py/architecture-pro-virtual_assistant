"""
Строит векторный индекс FAISS по документам knowledge_base/.

Документы делятся на чанки RecursiveCharacterTextSplitter по 300 слов
с перекрытием 50 слов. К каждому чанку добавляется заголовок документа,
чтобы чанк из середины статьи не терял, о ком или о чём идёт речь.
В метаданных чанка хранятся путь к файлу, заголовок, id чанка, его
номер в документе и позиция начала в файле (символ).

Эмбеддинги считает intfloat/multilingual-e5-small. Модель требует
префиксов: "passage: " для документов и "query: " для запросов.
Индекс сохраняется в faiss_index/ (index.faiss и index.pkl).

Использование: python build_index.py
"""

import logging
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB_DIR = Path("knowledge_base")
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_WORDS = 300
OVERLAP_WORDS = 50

log = logging.getLogger("build_index")


def embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True},
        query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True},
    )


def load_index() -> FAISS:
    # index.pkl - pickle, загружаем только индекс, собранный здесь
    return FAISS.load_local(str(INDEX_DIR), embeddings(), allow_dangerous_deserialization=True)


def word_count(text: str) -> int:
    return len(text.split())


def split_file(path: Path, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    heading, _, body = path.read_text(encoding="utf-8").partition("\n")
    title = heading.removeprefix("# ")
    offset = len(heading) + 1
    chunks = []
    for number, part in enumerate(splitter.create_documents([body])):
        metadata = {
            "source": str(path),
            "title": title,
            "chunk_id": f"{path.stem}:{number}",
            "chunk": number,
            "start_index": offset + part.metadata["start_index"],
        }
        chunks.append(Document(page_content=f"{title}\n{part.page_content}", metadata=metadata))
    return chunks


def split_documents(paths: list[Path]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_WORDS,
        chunk_overlap=OVERLAP_WORDS,
        length_function=word_count,
        add_start_index=True,
    )
    return [chunk for path in paths for chunk in split_file(path, splitter)]


def main() -> None:
    logging.basicConfig(format="%(message)s")
    log.setLevel(logging.INFO)
    paths = sorted(KB_DIR.glob("*.md"))
    chunks = split_documents(paths)
    log.info("документов: %d, чанков: %d", len(paths), len(chunks))

    started = time.perf_counter()
    model = embeddings()
    loaded = time.perf_counter()
    index = FAISS.from_documents(chunks, model, ids=[c.metadata["chunk_id"] for c in chunks])
    built = time.perf_counter()
    index.save_local(str(INDEX_DIR))

    size = sum(f.stat().st_size for f in INDEX_DIR.iterdir())
    log.info("модель: %s, размерность: %d", EMBEDDING_MODEL, index.index.d)
    log.info("загрузка модели: %.1f с", loaded - started)
    log.info("эмбеддинги и индекс: %.1f с", built - loaded)
    log.info("векторов: %d, размер %s: %d КБ", index.index.ntotal, INDEX_DIR, size // 1024)


if __name__ == "__main__":
    main()
