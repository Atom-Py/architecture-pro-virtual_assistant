"""
Автоматическая проверка RAG-бота на золотом наборе вопросов.

Вопросы берутся из golden_questions.txt: вопрос | ожидаемый ответ |
ключевые слова | ожидаемые источники. Каждый вопрос задаётся через
Rag.ask, поэтому ответы попадают и в журнал logs/logs.jsonl. Оценка:
- ответ найден: бот ответил не "Я не знаю";
- полнота: доля ключевых слов, которые есть в ответе;
- источник: среди источников ответа есть ожидаемый документ;
- итог: для вопроса с ответом в базе нужен найденный ответ с полнотой
  не ниже MIN_COMPLETENESS, для вопроса без ответа нужно "Я не знаю".
Ошибка LLM на вопросе засчитывается как неверный итог, проверка
продолжается.

Таблица результатов пишется в Markdown (по умолчанию logs/evaluation.md)
и в stdout. Если доля верных итогов ниже MIN_ACCURACY, скрипт
завершается с кодом 1, поэтому его можно запускать проверкой после
обновления индекса.

Использование: python evaluate.py [файл отчёта]
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from logging import WARNING, basicConfig
from pathlib import Path
from sys import argv, stdout
from sys import exit as sys_exit

from openai import APIError

from rag import Answer, Rag

GOLDEN = Path("golden_questions.txt")
REPORT = Path("logs/evaluation.md")
MIN_COMPLETENESS = 0.5
MIN_ACCURACY = 0.8


@dataclass
class Case:
    question: str
    expected: str
    keywords: list[str]
    sources: list[str]


@dataclass
class Result:
    case: Case
    answer: Answer
    completeness: float | None
    documents: list[str]
    correct: bool


def split_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def load_cases(path: Path) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        question, expected, keywords, sources = (part.strip() for part in line.split("|"))
        cases.append(Case(question, expected, split_list(keywords), split_list(sources)))
    return cases


def ask(rag: Rag, question: str) -> Answer:
    try:
        return rag.ask(question)
    except APIError as error:
        return Answer(f"Ответ: ошибка LLM ({type(error).__name__})")


def evaluate(case: Case, answer: Answer) -> Result:
    # источник в ответе выглядит как "<файл>, чанк <номер>"
    documents = sorted({source.split(",")[0] for source in answer.sources})
    if not case.keywords:
        return Result(case, answer, None, documents, correct=not answer.known)
    text = answer.text.lower()
    found = [keyword for keyword in case.keywords if keyword.lower() in text]
    completeness = len(found) / len(case.keywords)
    correct = answer.known and completeness >= MIN_COMPLETENESS
    return Result(case, answer, completeness, documents, correct)


def source_cell(result: Result) -> str:
    if not result.answer.known:
        return "-"
    if not result.case.keywords:
        return "ответ по: " + ", ".join(result.documents)
    if set(result.documents) & set(result.case.sources):
        return "верный"
    return "нерелевантный: " + ", ".join(result.documents)


def final_answer(answer: Answer) -> str:
    line = answer.text.rsplit("Ответ:", 1)[-1].strip().replace("\n", " ").replace("|", "/")
    return line if len(line) <= 120 else line[:117] + "..."


def report(results: list[Result], vectors: int) -> str:
    correct = sum(result.correct for result in results)
    with_answer = [result for result in results if result.case.keywords]
    without_answer = [result for result in results if not result.case.keywords]
    completeness = [result.completeness or 0.0 for result in with_answer]
    total = f"Индекс: {vectors} векторов. Верно: {correct} из {len(results)}"
    total += f" ({correct / len(results):.0%})."
    groups = f"Вопросы с ответом в базе: {sum(r.correct for r in with_answer)}"
    groups += f" из {len(with_answer)}, средняя полнота"
    groups += f" {sum(completeness) / len(completeness):.2f}."
    groups += f" Вопросы без ответа: {sum(r.correct for r in without_answer)}"
    groups += f" из {len(without_answer)}."
    lines = [
        f"# Оценка на золотом наборе, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        total,
        groups,
        "",
        "| № | Вопрос | Ожидается | Ответ найден | Полнота | Источник | Итог | Ответ бота |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for number, result in enumerate(results, 1):
        expected = "ответ" if result.case.keywords else "Я не знаю"
        completeness_cell = "-" if result.completeness is None else f"{result.completeness:.2f}"
        cells = [
            str(number),
            result.case.question,
            expected,
            "да" if result.answer.known else "нет",
            completeness_cell,
            source_cell(result),
            "верно" if result.correct else "ошибка",
            final_answer(result.answer),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    # httpx пишет адрес LLM на уровне INFO, поэтому только WARNING
    basicConfig(level=WARNING, format="%(message)s")
    path = Path(argv[1]) if len(argv) > 1 else REPORT
    rag = Rag()
    results = [evaluate(case, ask(rag, case.question)) for case in load_cases(GOLDEN)]
    text = report(results, rag.index.index.ntotal)
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    stdout.write(text)
    if sum(result.correct for result in results) / len(results) < MIN_ACCURACY:
        sys_exit(1)


if __name__ == "__main__":
    main()
