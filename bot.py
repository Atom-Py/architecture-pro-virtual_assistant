"""
Telegram-бот поверх RAG из rag.py.

Работает через long polling. Отвечает только пользователям, чьи
Telegram ID перечислены через запятую в TELEGRAM_ALLOWED_USERS,
сообщения остальных игнорируются. Токен и список пользователей
берутся из .env.

Использование: python bot.py
"""

from asyncio import run, to_thread
from logging import INFO, WARNING, basicConfig, getLogger
from os import environ

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import APIError

from rag import Rag

log = getLogger("bot")


async def main() -> None:
    # httpx пишет адрес LLM на уровне INFO, поэтому только WARNING
    basicConfig(level=WARNING, format="%(asctime)s %(name)s %(message)s")
    log.setLevel(INFO)
    getLogger("rag").setLevel(INFO)
    load_dotenv()
    allowed = {int(user) for user in environ["TELEGRAM_ALLOWED_USERS"].split(",")}
    rag = Rag()

    dispatcher = Dispatcher()
    dispatcher.message.filter(F.from_user.id.in_(allowed))

    @dispatcher.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer("Задайте вопрос по базе знаний.")

    @dispatcher.message(F.text)
    async def question(message: Message) -> None:
        try:
            answer = await to_thread(rag.ask, message.text)
        except APIError as error:
            log.warning("LLM не ответила: %s", type(error).__name__)
            await message.answer("Не удалось получить ответ, попробуйте позже.")
            return
        await message.answer(answer.render())

    log.info("бот запущен")
    await dispatcher.start_polling(Bot(environ["TELEGRAM_BOT_TOKEN"]))


if __name__ == "__main__":
    run(main())
