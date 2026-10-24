"""
Обновление индекса FAISS по папке knowledge_base/ по расписанию.

Новые, изменённые и удалённые документы находятся сравнением SHA-256
файлов с faiss_index/manifest.json последней сборки. Чанки изменённых
и удалённых документов удаляются из индекса по id (<имя файла>:<номер>),
новые и изменённые документы режутся на чанки и получают эмбеддинги
так же, как в build_index.py. Индекс сохраняется через save_index:
сначала во временный каталог, затем подменой файлов.

При ошибке обновление повторяется до ATTEMPTS раз с паузой RETRY_PAUSE
секунд. Если все попытки неудачны, индекс остаётся прежним, а скрипт
завершается с кодом 1. Ход работы пишется в logs/update.log и в stdout.

Использование: python update_index.py
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from json import loads
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path
from sys import exit as sys_exit
from sys import stdout
from time import gmtime, perf_counter, sleep

from langchain_core.documents import Document

from build_index import (
    INDEX_DIR,
    KB_DIR,
    MANIFEST,
    file_hash,
    load_index,
    save_index,
    split_documents,
)

LOG_FILE = Path("logs/update.log")
ATTEMPTS = 3
RETRY_PAUSE = 60

log = getLogger("update_index")


@dataclass
class Stats:
    added: int = 0
    changed: int = 0
    deleted: int = 0
    new_chunks: int = 0
    removed_chunks: int = 0
    vectors: int = 0


def update() -> Stats:
    hashes = {path.name: file_hash(path) for path in sorted(KB_DIR.glob("*.md"))}
    previous = loads(MANIFEST.read_text(encoding="utf-8"))
    added = [name for name in hashes if name not in previous]
    changed = [name for name in hashes if name in previous and hashes[name] != previous[name]]
    deleted = [name for name in previous if name not in hashes]
    index = load_index()
    stats = Stats(len(added), len(changed), len(deleted), vectors=index.index.ntotal)
    if not (added or changed or deleted):
        return stats
    log.info("новые: %s, изменённые: %s, удалённые: %s", added, changed, deleted)

    outdated = set(changed + deleted)
    stale = []
    for chunk_id in index.index_to_docstore_id.values():
        doc = index.docstore.search(chunk_id)
        if isinstance(doc, Document) and Path(doc.metadata["source"]).name in outdated:
            stale.append(chunk_id)
    if stale:
        index.delete(stale)
    chunks = split_documents([KB_DIR / name for name in added + changed])
    if chunks:
        index.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])
    save_index(index, hashes)

    stats.new_chunks = len(chunks)
    stats.removed_chunks = len(stale)
    stats.vectors = index.index.ntotal
    return stats


def main() -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    # время в UTC, как и расписание в crontab
    formatter = Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = gmtime
    for handler in (StreamHandler(stdout), FileHandler(LOG_FILE, encoding="utf-8")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(INFO)

    started = perf_counter()
    log.info("обновление индекса запущено")
    errors = 0
    for attempt in range(1, ATTEMPTS + 1):
        try:
            stats = update()
            break
        except Exception:
            errors += 1
            log.exception("попытка %d из %d не удалась", attempt, ATTEMPTS)
            if attempt < ATTEMPTS:
                sleep(RETRY_PAUSE)
    else:
        log.error("index not updated at %s, %d errors", f"{datetime.now(UTC):%Y-%m-%d}", errors)
        sys_exit(1)

    size = sum(f.stat().st_size for f in INDEX_DIR.iterdir() if f.is_file()) // 1024
    log.info(
        "index updated at %s, %d files added, %d changed, %d deleted, %d new chunks, "
        "%d removed chunks, %d vectors, %d KB, %d errors, %.1f s",
        f"{datetime.now(UTC):%Y-%m-%d}",
        stats.added,
        stats.changed,
        stats.deleted,
        stats.new_chunks,
        stats.removed_chunks,
        stats.vectors,
        size,
        errors,
        perf_counter() - started,
    )


if __name__ == "__main__":
    main()
