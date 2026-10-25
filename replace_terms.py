"""
Заменяет термины Star Wars на вымышленные по словарю terms_map.json.

Все термины ищутся одним регулярным выражением, более длинные раньше
коротких: "Darth Vader" заменяется целиком, а не по словам. Термин
совпадает только как целое слово с учётом регистра, поэтому "Force"
заменяется, а "force" нет. Притяжательная форма ('s) сохраняется.
Артикль перед термином подбирается под новое слово: "a droid"
становится "an automaton".

После замены текст проверяется: если термин остался в другом регистре
(например, "Droids" в начале предложения при термине "droids"),
скрипт выводит его и завершается с ошибкой.

Использование: python replace_terms.py [исходный каталог] [каталог базы]
               (по умолчанию raw/ и knowledge_base/)
"""

import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("replace_terms")


def build_pattern(terms: dict[str, str]) -> re.Pattern[str]:
    keys = sorted(terms, key=len, reverse=True)
    alternatives = "|".join(map(re.escape, keys))
    return re.compile(rf"(?<!\w)(?:(?P<article>[Aa]n?) )?(?P<term>{alternatives})(?!\w)")


def replace(match: re.Match[str], terms: dict[str, str]) -> str:
    value = terms[match["term"]]
    article = match["article"]
    if article is None:
        return value
    new = "an" if value[0].lower() in "aeiou" else "a"
    if article[0].isupper():
        new = new.capitalize()
    return f"{new} {value}"


def leftovers(text: str, terms: dict[str, str]) -> list[str]:
    found = []
    for key in terms:
        for match in re.finditer(rf"(?<!\w){re.escape(key)}(?!\w)", text, re.IGNORECASE):
            word = match.group(0)
            # "force" при термине "Force" - обычное слово, не термин
            if key[0].isupper() and word[0].islower():
                continue
            found.append(word)
    return found


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "knowledge_base")
    terms = json.loads(Path("terms_map.json").read_text(encoding="utf-8"))
    pattern = build_pattern(terms)
    dst.mkdir(parents=True, exist_ok=True)
    errors = 0
    for path in sorted(src.glob("*.md")):
        text, count = pattern.subn(lambda m: replace(m, terms), path.read_text(encoding="utf-8"))
        title = text.splitlines()[0].removeprefix("# ")
        out = dst / f"{slug(title)}.md"
        out.write_text(text, encoding="utf-8")
        log.info("%s -> %s: %d замен", path.name, out.name, count)
        rest = leftovers(text, terms)
        if rest:
            errors += 1
            log.error("%s: остались термины %s", out.name, sorted(set(rest)))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
