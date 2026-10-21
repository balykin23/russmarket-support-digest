#!/usr/bin/env python3
"""Daily support digest: CRM statuses + work-chat messages -> prioritized to-do list.

Design in one line: deterministic rules decide everything that can be checked
(dates, statuses, deadlines); the LLM only reads chat messages and extracts
what is *claimed*. A chat message never closes a task: only CRM confirms.

Usage:
    python3 summary.py data/cases.json                 # mock: recorded LLM answers, no key needed
    python3 summary.py data/cases.json --mode live     # real OpenAI call (OPENAI_API_KEY in .env)
    python3 summary.py data/cases.json --mode live --record   # also save answers to fixtures/
    python3 summary.py data/cases.json --send          # also post the digest to Telegram

Standard library only (Python 3.9+), no pip install required.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures" / "llm_responses.json"
DEFAULT_MODEL = "gpt-5.4-mini"

# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------
CRITICAL, HIGH, CHECK, OK = 0, 1, 2, 3
LEVEL_LABEL = {
    CRITICAL: "🔴 Срочно",
    HIGH: "🟠 До конца дня",
    CHECK: "🟡 Проверить",
    OK: "🟢 Действий нет",
}
# Tie-break inside one level: the less reversible the consequence, the earlier.
# Offboarding: final settlement and documents are due on the last working day.
# Card delivery: the point is waiting, but the delay itself is recoverable.
# Onboarding: a first day can be moved if documents are late.
TYPE_RANK = {"offboarding": 0, "card_delivery": 1, "onboarding": 2}
TYPE_REASON = {
    "offboarding": "при равной срочности увольнение идёт первым: расчёт и документы "
                   "выдаются в последний рабочий день, ошибку потом не исправить",
    "card_delivery": "при равной срочности доставка идёт раньше оформления: точка ждёт карты сейчас",
}

# ---------------------------------------------------------------------------
# LLM contract: extraction only
# ---------------------------------------------------------------------------
CLAIMS = [
    "documents_sent",     # employee says documents were sent / handed over
    "manager_informed",   # someone says the manager knows / agreed
    "delivery_delayed",   # courier or anyone reports a delay / new arrival time
    "delivery_received",  # recipient says cards arrived
    "problem_reported",   # shortage, damage, wrong address, refusal, any problem
    "no_info",            # greetings, thanks, nothing actionable
    "other",              # something relevant that fits none of the above
]

SYSTEM_PROMPT = """\
Ты разбираешь сообщения из рабочих чатов support-отдела. Отдел следит за
оформлением и увольнением сотрудников и за доставкой банковских карт на точки.

Твоя задача — только извлечь, ЧТО утверждается в каждом сообщении.
Не оценивай, правда ли это. Не решай, нужно ли вмешиваться. Не ставь приоритет.
Не додумывай факты, которых нет в тексте.

Для каждого сообщения верни:
- index — номер сообщения из входных данных;
- claim — одно значение:
  documents_sent — сотрудник говорит, что УЖЕ отправил/передал документы
    (обещание передать позже — это other);
  manager_informed — кто-то говорит, что руководитель в курсе/согласовал;
  delivery_delayed — сообщают о задержке доставки или новом времени прибытия;
  delivery_received — получатель говорит, что карты пришли;
  problem_reported — сообщают о проблеме: недостача, брак, не тот адрес, отказ и т.п.;
  no_info — приветствие, благодарность, ничего по делу;
  other — что-то важное, что не подходит под варианты выше;
- eta — новое время прибытия в формате HH:MM, если оно названо; иначе null.
  «приеду не раньше 15:30» → "15:30"; «буду к четырём часам дня» → "16:00";
- unofficial_channel — true, если сказано, что документы или данные переданы
  в обход рабочего процесса (личные сообщения, личная почта, «скинул в личку»);
  иначе false.

Если в сообщении есть несколько утверждений, выбери главное для support-отдела:
проблема важнее подтверждения получения.
Ответ — строго JSON по схеме, без пояснений."""

RESPONSE_SCHEMA = {
    "name": "chat_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["messages"],
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "claim", "eta", "unofficial_channel"],
                    "properties": {
                        "index": {"type": "integer"},
                        "claim": {"type": "string", "enum": CLAIMS},
                        "eta": {"type": ["string", "null"]},
                        "unofficial_channel": {"type": "boolean"},
                    },
                },
            }
        },
    },
}

# ---------------------------------------------------------------------------
# PII masking: applied before any text leaves for the LLM
# ---------------------------------------------------------------------------
PII_PATTERNS = [
    ("[КАРТА]", re.compile(r"\b(?:\d[ -]?){15}\d\b")),
    ("[ТЕЛЕФОН]", re.compile(r"(?:\+7|\b8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b")),
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("[ПАСПОРТ]", re.compile(r"\b\d{2}\s?\d{2}\s?\d{6}\b")),
]


def mask_pii(text: str) -> str:
    for token, pattern in PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


# ---------------------------------------------------------------------------
# Extraction backends
# ---------------------------------------------------------------------------
class ExtractionError(Exception):
    pass


def build_user_payload(case: dict) -> dict:
    return {
        "case_type": case["type"],
        "messages": [
            {"index": i, "author_role": m["author_role"], "text": mask_pii(m["text"])}
            for i, m in enumerate(case.get("chat", []))
        ],
    }


def validate_extraction(raw, n_messages: int) -> list[dict]:
    """Never trust model output: check shape, enums and formats."""
    if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
        raise ExtractionError("ответ модели не соответствует схеме")
    seen, result = set(), []
    for item in raw["messages"]:
        if not isinstance(item, dict):
            raise ExtractionError("элемент ответа не объект")
        idx = item.get("index")
        if not isinstance(idx, int) or not 0 <= idx < n_messages or idx in seen:
            raise ExtractionError(f"неверный index: {idx!r}")
        if item.get("claim") not in CLAIMS:
            raise ExtractionError(f"неизвестный claim: {item.get('claim')!r}")
        eta = item.get("eta")
        if eta is not None and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(eta)):
            raise ExtractionError(f"неверный формат eta: {eta!r}")
        seen.add(idx)
        result.append(item)
    if len(seen) != n_messages:
        raise ExtractionError("модель разобрала не все сообщения")
    return sorted(result, key=lambda x: x["index"])


def call_openai(payload: dict, model: str) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ExtractionError("OPENAI_API_KEY не задан")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:  # never print the request: it carries the key
        print(f"[warn] OpenAI HTTP {e.code}", file=sys.stderr)
        raise ExtractionError(f"модель ответила ошибкой {e.code}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"[warn] OpenAI недоступен: {getattr(e, 'reason', e)}", file=sys.stderr)
        raise ExtractionError("модель недоступна") from None
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ExtractionError("модель вернула не JSON") from None


class Extractor:
    def __init__(self, mode: str, model: str, record: bool):
        self.mode, self.model, self.record = mode, model, record
        self.fixtures = json.loads(FIXTURES.read_text("utf-8")) if FIXTURES.exists() else {}
        self.recorded: dict = {}

    def extract(self, case: dict) -> list[dict]:
        n = len(case.get("chat", []))
        if n == 0:
            return []
        if self.mode == "mock":
            if case["id"] not in self.fixtures:
                raise ExtractionError("нет записанного ответа модели")
            raw = self.fixtures[case["id"]]
        else:
            raw = call_openai(build_user_payload(case), self.model)
            if self.record:
                self.recorded[case["id"]] = raw
        return validate_extraction(raw, n)

    def save(self) -> None:
        if self.record and self.recorded:
            meta = dict(self.fixtures.get("_meta", {}))
            meta.update({cid: {"model": self.model, "recorded_at": datetime.now().isoformat(timespec="seconds")}
                         for cid in self.recorded})
            merged = {**self.fixtures, **self.recorded, "_meta": meta}
            FIXTURES.parent.mkdir(exist_ok=True)
            FIXTURES.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", "utf-8")


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
@dataclass
class Assessment:
    case: dict
    level: int = OK
    why: str = ""
    crm: list[str] = field(default_factory=list)      # confirmed by CRM
    chat: list[str] = field(default_factory=list)     # known only from chat
    check: list[str] = field(default_factory=list)    # what to verify before acting
    actions: list[str] = field(default_factory=list)  # what to do next
    flags: list[str] = field(default_factory=list)    # process / compliance notes

    def raise_to(self, level: int, why: str) -> None:
        if level < self.level:
            self.level, self.why = level, why

    @property
    def needs_action(self) -> bool:
        return self.level != OK


def rel_day(d: date, today: date) -> str:
    delta = (d - today).days
    return {0: "сегодня", 1: "завтра", -1: "вчера"}.get(
        delta, f"через {delta} дн." if delta > 0 else f"{-delta} дн. назад"
    )


def fmt_dt(dt: datetime, now: datetime) -> str:
    if dt.date() == now.date():
        return f"сегодня {dt:%H:%M}"
    return f"{dt:%d.%m %H:%M}"


def level_by_days(days_left: int) -> int:
    if days_left <= 0:
        return CRITICAL
    if days_left == 1:
        return HIGH
    if days_left <= 3:
        return CHECK
    return OK


def assess_onboarding(a: Assessment, crm: dict, today: date) -> None:
    first_day = date.fromisoformat(crm["first_day"])
    days = (first_day - today).days
    a.crm.append(f"Первый выход: {first_day:%d.%m} ({rel_day(first_day, today)})")
    if crm.get("documents_received"):
        a.crm.append("Документы получены")
        return
    a.crm.append("Документы НЕ получены")
    lvl = level_by_days(days)
    if days < 0:
        a.raise_to(lvl, "первый выход уже прошёл, а документов в CRM нет")
        a.actions.append("Выяснить у руководителя, вышел ли сотрудник и на каком основании допущен без документов")
    elif lvl != OK:
        a.raise_to(lvl, f"первый выход {rel_day(first_day, today)}, документов в CRM нет")
    a.check.append("Где документы сейчас и все ли они читаемы")
    a.actions.append("Получить документы в рабочем канале и отметить получение в CRM")


def assess_offboarding(a: Assessment, crm: dict, today: date) -> None:
    last_day = date.fromisoformat(crm["last_day"])
    days = (last_day - today).days
    a.crm.append(f"Последний рабочий день: {last_day:%d.%m} ({rel_day(last_day, today)})")
    if crm.get("manager_confirmed"):
        a.crm.append("Руководитель подтвердил")
        return
    a.crm.append("Подтверждения руководителя НЕТ")
    lvl = level_by_days(days)
    if lvl == OK:
        a.why = f"увольнение {rel_day(last_day, today)} — подтверждение пока не горит"
        return
    a.raise_to(lvl, f"последний день {rel_day(last_day, today)}, руководитель не подтвердил")
    a.actions.append("Написать руководителю напрямую и получить подтверждение в CRM")
    if days <= 0:
        a.actions.append("Сегодня же пройти чек-лист увольнения: расчёт, документы, возврат имущества и доступов")


def assess_delivery(a: Assessment, crm: dict, now: datetime) -> None:
    deadline = datetime.fromisoformat(crm["deadline"])
    status = crm.get("status", "")
    accepted_at = crm.get("accepted_at")
    if status == "Принято" and accepted_at:
        a.crm.append(f"Статус «Принято», приёмка {fmt_dt(datetime.fromisoformat(accepted_at), now)}")
        return
    a.crm.append(f"Статус «{status}», срок доставки {fmt_dt(deadline, now)}")
    if now >= deadline:  # "до 10:00" and it is 10:00 with no acceptance -> overdue
        late = int((now - deadline).total_seconds() // 60)
        if late == 0:
            a.crm.append("Срок наступил, приёмки нет")
        elif late < 60:
            a.crm.append(f"Срок прошёл {late} мин назад, приёмки нет")
        else:
            a.crm.append(f"Срок прошёл {late // 60} ч {late % 60} мин назад, приёмки нет")
        a.raise_to(CRITICAL, "срок доставки наступил, в CRM карты не приняты")


def apply_chat(a: Assessment, items: list[dict], now: datetime) -> None:
    ctype, crm = a.case["type"], a.case["crm"]
    msgs = a.case.get("chat", [])
    for item in items:
        role = ROLE_RU.get(msgs[item["index"]]["author_role"], msgs[item["index"]]["author_role"])
        # Quote the original message instead of an LLM paraphrase: a live run showed the model
        # strengthening claims ("он в курсе" -> "всё согласовано"). The model only classifies.
        text = msgs[item["index"]]["text"]
        summary = "«" + (text if len(text) <= 200 else text[:197] + "...") + "»"
        claim = item["claim"]
        if claim == "no_info":
            continue

        if item.get("unofficial_channel"):
            a.flags.append("Документы с персональными данными ушли в личные сообщения — "
                           "перевести в рабочий канал, из лички удалить")

        if claim == "documents_sent" and ctype == "onboarding" and not crm.get("documents_received"):
            a.chat.append(f"{role}: {summary} — в CRM не подтверждено")
            a.check.append("Подтвердить у руководителя, что документы действительно получены")
        elif claim == "manager_informed" and ctype == "offboarding" and not crm.get("manager_confirmed"):
            a.chat.append(f"{role}: {summary} — это слова сотрудника, не подтверждение")
            a.check.append("Слова сотрудника не заменяют подтверждение — спросить руководителя напрямую")
        elif claim == "delivery_received":
            if crm.get("status") == "Принято":
                a.chat.append(f"{role}: {summary} — совпадает с CRM")
            else:
                a.chat.append(f"{role}: {summary} — в CRM приёмки нет")
                a.check.append("Подтвердить у точки получение и количество")
                a.actions.insert(0, "Попросить точку провести приёмку в CRM")
        elif claim == "delivery_delayed":
            eta_txt = f" (новое время {item['eta']})" if item.get("eta") and item["eta"] not in summary else ""
            a.chat.append(f"{role}: {summary}{eta_txt} — в CRM не отражено")
            if item.get("eta"):
                h, m = map(int, item["eta"].split(":"))
                eta = now.replace(hour=h, minute=m, second=0, microsecond=0)
                deadline = datetime.fromisoformat(crm["deadline"])
                if eta > deadline:
                    if now < deadline:
                        a.raise_to(HIGH, f"курьер прогнозирует опоздание: {item['eta']} при сроке {deadline:%H:%M}")
                    a.actions.insert(0, f"Предупредить точку: карты будут не раньше {item['eta']}")
                a.actions.append(f"Обновить срок в CRM и проверить статус после {item['eta']}")
            else:
                a.actions.append("Узнать у курьера точное время прибытия и внести в CRM")
        elif claim == "problem_reported":
            a.chat.append(f"{role}: {summary} — в CRM не отражено")
            a.raise_to(HIGH, "в CRM всё в порядке, но в переписке сообщают о проблеме")
            a.check.append("Связаться с автором сообщения и подтвердить проблему")
            a.actions.append("Если подтвердится — зафиксировать расхождение в CRM")
        else:
            a.chat.append(f"{role}: {summary}")
            a.raise_to(CHECK, "в переписке есть сообщение, которое правила не покрывают")
            a.check.append("Прочитать сообщение и решить, нужно ли действие")


ROLE_RU = {"employee": "Сотрудник", "courier": "Курьер", "recipient": "Получатель", "manager": "Руководитель"}


def assess(case: dict, now: datetime, extractor: Extractor) -> Assessment:
    a = Assessment(case=case)
    crm = case["crm"]
    if case["type"] == "onboarding":
        assess_onboarding(a, crm, now.date())
    elif case["type"] == "offboarding":
        assess_offboarding(a, crm, now.date())
    elif case["type"] == "card_delivery":
        assess_delivery(a, crm, now)
    else:
        a.raise_to(CHECK, f"неизвестный тип операции «{case['type']}»")

    try:
        items = extractor.extract(case)
    except ExtractionError as e:  # fail-safe: a case is never dropped
        a.chat.append(f"⚠ Переписку не удалось разобрать автоматически ({e})")
        a.check.append("Прочитать переписку по кейсу вручную")
        a.raise_to(CHECK, "переписку не удалось разобрать автоматически")
        return a
    apply_chat(a, items, now)
    if case["type"] == "card_delivery" and a.level == CRITICAL and not any(
        i["claim"] in ("delivery_delayed", "delivery_received") for i in items
    ):
        a.actions.append("Узнать у курьера, где карты и когда будут, внести в CRM")
        a.actions.append("Предупредить точку о задержке")
    if not case.get("chat") and a.needs_action:
        a.chat.append("Переписки по кейсу нет")
    # de-duplicate while keeping order
    for attr in ("check", "actions", "flags"):
        setattr(a, attr, list(dict.fromkeys(getattr(a, attr))))
    return a


def sort_key(a: Assessment):
    return (a.level, TYPE_RANK.get(a.case["type"], 9), a.case["id"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


def render_markdown(results: list[Assessment], now: datetime) -> str:
    todo = [r for r in results if r.needs_action]
    out = [f"# Сводка support-отдела — {WEEKDAYS[now.weekday()]}, {now:%d.%m.%Y %H:%M}", ""]
    out.append(f"Требуют действий: **{len(todo)}** из {len(results)}. "
               "✅ — подтверждено в CRM, 💬 — известно только из переписки.")
    out.append("")
    if todo:
        first = todo[0]
        out.append(f"**Начать с {first.case['id']}** — {first.why}.")
        if len(todo) > 1 and todo[1].level == first.level and first.case["type"] in TYPE_REASON:
            out.append(f"Почему раньше {todo[1].case['id']}: {TYPE_REASON[first.case['type']]}.")
        out.append("")
        out.append("Порядок: " + " → ".join(r.case["id"] for r in todo))
        out.append("")
    for r in results:
        c = r.case
        out.append(f"## {LEVEL_LABEL[r.level]} · {c['id']} · {c['title']}")
        out.append(f"**Вмешаться:** {'да — ' + r.why if r.needs_action else 'нет' + (' — ' + r.why if r.why else '')}")
        out.append("")
        for line in r.crm:
            out.append(f"- ✅ {line}")
        for line in r.chat:
            out.append(f"- 💬 {line}")
        if r.check:
            out.append("")
            out.append("**Проверить перед действием:**")
            out.extend(f"- {x}" for x in r.check)
        if r.actions:
            out.append("")
            out.append("**Дальше:**")
            out.extend(f"{i}. {x}" for i, x in enumerate(r.actions, 1))
        if r.flags:
            out.append("")
            out.extend(f"> ⚠ {x}" for x in r.flags)
        out.append("")
    return "\n".join(out)


def render_telegram(results: list[Assessment], now: datetime) -> str:
    e = html.escape
    todo = [r for r in results if r.needs_action]
    lines = [f"<b>Сводка support · {WEEKDAYS[now.weekday()]} {now:%H:%M}</b>",
             f"Требуют действий: {len(todo)} из {len(results)}"]
    if todo:
        lines.append(f"▶️ Начать с <b>{e(todo[0].case['id'])}</b> — {e(todo[0].why)}")
    for r in results:
        lines.append("")
        if not r.needs_action:  # keep the "all good" cases to one line: no noise
            lines.append(f"{LEVEL_LABEL[r.level]} <b>{e(r.case['id'])}</b> · {e(r.case['title'])}: "
                         + e("; ".join(r.crm)))
            continue
        lines.append(f"{LEVEL_LABEL[r.level]} <b>{e(r.case['id'])}</b> · {e(r.case['title'])}")
        lines.extend(f"✅ {e(x)}" for x in r.crm)
        lines.extend(f"💬 {e(mask_pii(x))}" for x in r.chat)  # Telegram is outside RU infra
        if r.check:
            lines.append("🔎 " + e("; ".join(r.check)))
        lines.extend(f"👉 {e(x)}" for x in r.actions)
        lines.extend(f"⚠️ {e(x)}" for x in r.flags)
    lines.append("")
    lines.append("<i>✅ — CRM, 💬 — только переписка. Перед действием сверь с CRM.</i>")
    return "\n".join(lines)


def send_telegram(text: str) -> None:
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise SystemExit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы")
    body = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                       "disable_web_page_preview": True}).encode("utf-8")
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if not json.load(resp).get("ok"):
                raise SystemExit("Telegram ответил ошибкой")
    except urllib.error.HTTPError as e:  # URL contains the token: print only Telegram's description
        try:
            desc = json.load(e).get("description", "")
        except Exception:
            desc = ""
        raise SystemExit(f"Telegram HTTP {e.code}: {desc}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise SystemExit(f"Telegram недоступен: {getattr(e, 'reason', e)}") from None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run(data: dict, extractor: Extractor) -> tuple[list[Assessment], datetime]:
    now = datetime.fromisoformat(data["now"]) if data.get("now") else datetime.now()
    results = sorted((assess(c, now, extractor) for c in data["cases"]), key=sort_key)
    return results, now


def main() -> None:
    p = argparse.ArgumentParser(description="Ежедневная сводка support-отдела")
    p.add_argument("input", help="JSON с кейсами (CRM + переписка)")
    p.add_argument("--mode", choices=["mock", "live"], default="mock")
    p.add_argument("--model", default=None, help=f"модель OpenAI (по умолчанию {DEFAULT_MODEL})")
    p.add_argument("--record", action="store_true", help="сохранить ответы модели в fixtures/")
    p.add_argument("--send", action="store_true", help="отправить сводку в Telegram")
    p.add_argument("--out", default=None, help="куда сохранить markdown (по умолчанию stdout)")
    args = p.parse_args()

    load_env(ROOT / ".env")
    if "SSL_CERT_FILE" not in os.environ:  # python.org builds on macOS ship without root certs
        try:
            import certifi  # optional: used only if installed
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass
    model = args.model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    data = json.loads(Path(args.input).read_text("utf-8"))
    extractor = Extractor(args.mode, model, args.record)
    results, now = run(data, extractor)
    extractor.save()

    md = render_markdown(results, now)
    if args.out:
        Path(args.out).write_text(md + "\n", "utf-8")
        print(f"Сводка сохранена: {args.out}")
    else:
        print(md)
    if args.send:
        send_telegram(render_telegram(results, now))
        print("Отправлено в Telegram")


if __name__ == "__main__":
    main()
