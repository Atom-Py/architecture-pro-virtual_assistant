"""
Скачивает начало статей Wookieepedia чистым текстом.

Статья берётся через MediaWiki API. Из HTML остаются только абзацы
верхнего уровня, без сносок, инфобоксов и цитат. Абзацы добавляются
по порядку, начиная с вводной части, пока текст не превысит MAX_WORDS
слов. Длинные тире заменяются дефисами.

Использование: python fetch_pages.py [каталог]  (по умолчанию raw/)
"""

import logging
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup
from requests import Session

API = "https://starwars.fandom.com/api.php"
MAX_WORDS = 450

PAGES = [
    # персонажи
    "Anakin Skywalker",
    "Luke Skywalker",
    "Leia Skywalker Organa Solo",
    "Han Solo",
    "Chewbacca",
    "Obi-Wan Kenobi",
    "Yoda",
    "Sheev Palpatine",
    "R2-D2",
    "Boba Fett",
    # планеты
    "Tatooine",
    "Coruscant",
    "Hoth",
    "Endor",
    "Naboo",
    "Alderaan",
    "Dagobah",
    # техника и технологии
    "DS-1 Orbital Battle Station",
    "Millennium Falcon",
    "Lightsaber",
    "T-65B X-wing starfighter",
    "TIE/ln space superiority starfighter",
    "Hyperdrive",
    # организации и понятия
    "Force",
    "Jedi Order",
    "Sith",
    "Galactic Empire",
    "Alliance to Restore the Republic",
    # события
    "Battle of Yavin",
    "Battle of Endor",
    "Clone Wars",
    # расы
    "Wookiee",
]

log = logging.getLogger("fetch_pages")


def fetch_page(session: Session, title: str) -> tuple[str, str]:
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1,
        "format": "json",
    }
    response = session.get(API, params=params, timeout=30)
    response.raise_for_status()
    parse = response.json()["parse"]
    return parse["title"], parse["text"]["*"]


def clean(html: str) -> list[str]:
    root = BeautifulSoup(html, "html.parser").select_one(".mw-parser-output")
    for sup in root.select("sup"):
        sup.decompose()
    paragraphs = []
    words = 0
    for p in root.find_all("p", recursive=False):
        text = " ".join(p.get_text().split())
        text = text.replace(" \u2014 ", " - ").replace("\u2014", " - ").replace("\u2013", "-")
        if not text:
            continue
        paragraphs.append(text)
        words += len(text.split())
        if words >= MAX_WORDS:
            break
    return paragraphs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    session = Session()
    session.headers["User-Agent"] = "rag-study-project/0.1"
    for title in PAGES:
        real_title, html = fetch_page(session, title)
        paragraphs = clean(html)
        name = real_title.replace("/", "-").replace(" ", "_") + ".md"
        text = f"# {real_title}\n\n" + "\n\n".join(paragraphs) + "\n"
        (out_dir / name).write_text(text, encoding="utf-8")
        log.info("%s: %d слов", real_title, len(text.split()))
        time.sleep(0.5)


if __name__ == "__main__":
    main()
