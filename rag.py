"""
RAG: поиск по индексу FAISS и ответ LLM с few-shot и Chain-of-Thought.

Цепочка для одного вопроса:
1. вопрос переводится в эмбеддинг той же моделью, что и индекс
   (multilingual-e5-small с префиксом "query: ");
2. в индексе ищутся TOP_K ближайших чанков;
3. чанки дальше MAX_DISTANCE отбрасываются; если не осталось ни
   одного, бот отвечает "Я не знаю." без обращения к LLM;
4. собирается промпт: системное сообщение с требованием писать шаги
   рассуждения, два примера вопрос-ответ по фрагментам базы (few-shot),
   найденные чанки с источниками и вопрос пользователя;
5. ответ LLM возвращается вместе с источниками: фрагментами, на которые
   модель сослалась в шагах ([1], [2]), а если ссылок нет, всеми.

LLM вызывается через OpenAI-совместимый API, адрес, ключ и модель
берутся из .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.

Использование: python rag.py "вопрос"
"""

from dataclasses import dataclass, field
from logging import WARNING, basicConfig
from os import environ
from pathlib import Path
from re import findall
from sys import argv, stdout

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from build_index import load_index

TOP_K = 4
MAX_DISTANCE = 0.5
DONT_KNOW = "Я не знаю."

SYSTEM_PROMPT = """\
Ты помощник по базе знаний. Отвечай только по фрагментам документов, \
которые приходят вместе с вопросом. Не используй знания, которых нет во \
фрагментах, и ничего не придумывай.

Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши \
свои шаги в таком виде:
Шаги:
1. ...
2. ...
Ответ: ...

Если во фрагментах нет ответа на вопрос, после шагов напиши: "Ответ: Я не знаю."
Ответ пиши на русском языке, по-английски оставляй только имена и названия \
из фрагментов."""

FEW_SHOT = [
    (
        """\
[1] dax_corvo.md, чанк 2
Dax Corvo
With only Grawrok remaining at his side, as his trusted copilot who swore a life debt to the \
Brannickan smuggler, Dax Corvo embarked on a smuggling career under the employ of Gorrba the \
Gruuk, a notorious gangster and crime lord of Drassa.""",
        "Кто был вторым пилотом Dax Corvo?",
        """\
Шаги:
1. Ищу во фрагментах, кто летал вместе с Dax Corvo.
2. Во фрагменте [1] сказано, что рядом с ним остался Grawrok, "his trusted copilot".
3. Значит, вторым пилотом был Grawrok.
Ответ: вторым пилотом Dax Corvo был Grawrok.""",
    ),
    (
        """\
[1] drassa.md, чанк 0
Drassa
Drassa was a sparsely inhabited circumbinary desert planet located in the galaxy's Far Fringe \
Territories. Part of a binary star system, the planet orbited two scorching suns, resulting in \
the world lacking the necessary surface water to sustain large populations.""",
        "Кто основал первое поселение на Drassa?",
        """\
Шаги:
1. Ищу во фрагментах сведения о первом поселении на Drassa.
2. Фрагмент [1] описывает климат и солнца Drassa, о поселениях и их основателях в нём ничего нет.
3. Ответа во фрагментах нет.
Ответ: Я не знаю.""",
    ),
]


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return "я не знаю" not in self.text.rsplit("Ответ:", 1)[-1].lower()

    def render(self) -> str:
        if not self.known or not self.sources:
            return self.text
        sources = "\n".join(f"- {source}" for source in self.sources)
        return f"{self.text}\n\nИсточники:\n{sources}"


def user_message(context: str, question: str) -> str:
    return f"Фрагменты документов:\n\n{context}\n\nВопрос: {question}"


def cited(text: str, chunks: list[Document]) -> list[Document]:
    numbers = {int(number) for number in findall(r"\[(\d+)\]", text)}
    return [doc for number, doc in enumerate(chunks, 1) if number in numbers] or chunks


def source_name(doc: Document) -> str:
    return f"{Path(doc.metadata['source']).name}, чанк {doc.metadata['chunk']}"


def build_messages(chunks: list[Document], question: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    for context, example_question, example_answer in FEW_SHOT:
        messages.append(HumanMessage(user_message(context, example_question)))
        messages.append(AIMessage(example_answer))
    context = "\n\n".join(
        f"[{number}] {source_name(doc)}\n{doc.page_content}"
        for number, doc in enumerate(chunks, 1)
    )
    messages.append(HumanMessage(user_message(context, question)))
    return messages


class Rag:
    def __init__(self) -> None:
        load_dotenv()
        self.index = load_index()
        self.llm = ChatOpenAI(
            base_url=environ["LLM_BASE_URL"],
            api_key=environ["LLM_API_KEY"],
            model=environ["LLM_MODEL"],
            temperature=0,
            timeout=120,
        )

    def ask(self, question: str) -> Answer:
        found = self.index.similarity_search_with_score(question, k=TOP_K)
        distances = [float(distance) for _, distance in found]
        chunks = [doc for doc, distance in found if distance <= MAX_DISTANCE]
        if not chunks:
            return Answer(DONT_KNOW, distances=distances)
        reply = self.llm.invoke(build_messages(chunks, question))
        text = str(reply.text)
        return Answer(text, [source_name(doc) for doc in cited(text, chunks)], distances)


def main() -> None:
    # httpx пишет адрес LLM на уровне INFO, поэтому только WARNING
    basicConfig(level=WARNING, format="%(message)s")
    answer = Rag().ask(" ".join(argv[1:]))
    stdout.write(answer.render() + "\n")


if __name__ == "__main__":
    main()
