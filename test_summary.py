"""Rule tests. Run: python3 -m unittest -v
The LLM is replaced by a stub, so the tests check the rules, not the model."""
import json
import unittest
from datetime import datetime
from pathlib import Path

import summary as s

ROOT = Path(__file__).resolve().parent
NOW = datetime(2026, 10, 19, 10, 0)


class StubExtractor:
    """Returns prepared answers; a non-dict answer simulates a broken model reply."""

    def __init__(self, answers):
        self.answers = answers

    def extract(self, case):
        n = len(case.get("chat", []))
        if n == 0:
            return []
        if case["id"] not in self.answers:
            raise s.ExtractionError("нет ответа")
        return s.validate_extraction(self.answers[case["id"]], n)


def msg(claim, eta=None, unofficial=False, index=0):
    return {"messages": [{"index": index, "claim": claim,
                          "eta": eta, "unofficial_channel": unofficial}]}


def delivery(status="В пути", deadline="2026-10-19T10:00:00", accepted_at=None, chat=None, cid="X"):
    return {"id": cid, "type": "card_delivery", "title": "t",
            "crm": {"status": status, "deadline": deadline, "accepted_at": accepted_at},
            "chat": chat or []}


class TestFourCases(unittest.TestCase):
    """The four cases from the assignment, with the recorded model answers."""

    def setUp(self):
        data = json.loads((ROOT / "data" / "cases.json").read_text("utf-8"))
        fixtures = json.loads((ROOT / "fixtures" / "llm_responses.json").read_text("utf-8"))
        self.results, _ = s.run(data, StubExtractor(fixtures))
        self.by_id = {r.case["id"]: r for r in self.results}

    def test_order(self):
        self.assertEqual([r.case["id"] for r in self.results], ["С-24", "П-08", "С-17", "П-09"])

    def test_levels(self):
        self.assertEqual(self.by_id["С-24"].level, s.CRITICAL)
        self.assertEqual(self.by_id["П-08"].level, s.CRITICAL)
        self.assertEqual(self.by_id["С-17"].level, s.HIGH)
        self.assertEqual(self.by_id["П-09"].level, s.OK)

    def test_chat_is_labelled_unconfirmed(self):
        self.assertTrue(any("не подтверждение" in x for x in self.by_id["С-24"].chat))
        self.assertTrue(any("не подтверждено" in x for x in self.by_id["С-17"].chat))
        self.assertTrue(any("не отражено" in x for x in self.by_id["П-08"].chat))

    def test_chat_is_quoted_verbatim(self):
        # the digest must show the original words, not a model paraphrase
        self.assertTrue(any("«С руководителем всё обсудили, он в курсе.»" in x for x in self.by_id["С-24"].chat))

    def test_personal_messages_flagged(self):
        self.assertTrue(self.by_id["С-17"].flags)

    def test_no_noise_when_crm_and_chat_agree(self):
        r = self.by_id["П-09"]
        self.assertEqual((r.check, r.actions, r.flags), ([], [], []))


class TestChatNeverClosesTask(unittest.TestCase):
    def test_manager_informed_does_not_lower_priority(self):
        case = {"id": "A", "type": "offboarding", "title": "t",
                "crm": {"last_day": "2026-10-19", "manager_confirmed": False},
                "chat": [{"author_role": "employee", "text": "он в курсе"}]}
        r = s.assess(case, NOW, StubExtractor({"A": msg("manager_informed")}))
        self.assertEqual(r.level, s.CRITICAL)

    def test_received_in_chat_but_not_in_crm_stays_critical(self):
        case = delivery(deadline="2026-10-19T09:00:00", cid="B",
                        chat=[{"author_role": "recipient", "text": "всё пришло"}])
        r = s.assess(case, NOW, StubExtractor({"B": msg("delivery_received")}))
        self.assertEqual(r.level, s.CRITICAL)
        self.assertIn("Попросить точку провести приёмку в CRM", r.actions)


class TestChatRaisesPriority(unittest.TestCase):
    def test_problem_reported_on_accepted_delivery(self):
        case = delivery(status="Принято", accepted_at="2026-10-19T08:50:00", cid="C",
                        chat=[{"author_role": "recipient", "text": "не хватает двух карт"}])
        r = s.assess(case, NOW, StubExtractor({"C": msg("problem_reported")}))
        self.assertEqual(r.level, s.HIGH)

    def test_eta_after_deadline_is_forecast_delay(self):
        case = delivery(deadline="2026-10-19T15:00:00", cid="D",
                        chat=[{"author_role": "courier", "text": "буду к 17"}])
        r = s.assess(case, NOW, StubExtractor({"D": msg("delivery_delayed", eta="17:00")}))
        self.assertEqual(r.level, s.HIGH)

    def test_eta_before_deadline_no_action(self):
        case = delivery(deadline="2026-10-19T15:00:00", cid="E",
                        chat=[{"author_role": "courier", "text": "буду к 12"}])
        r = s.assess(case, NOW, StubExtractor({"E": msg("delivery_delayed", eta="12:00")}))
        self.assertEqual(r.level, s.OK)


class TestDeadlines(unittest.TestCase):
    def test_delivery_before_deadline_is_ok(self):
        r = s.assess(delivery(deadline="2026-10-19T12:00:00"), NOW, StubExtractor({}))
        self.assertEqual(r.level, s.OK)

    def test_delivery_exactly_at_deadline_is_overdue(self):
        r = s.assess(delivery(deadline="2026-10-19T10:00:00"), NOW, StubExtractor({}))
        self.assertEqual(r.level, s.CRITICAL)
        self.assertIn("Узнать у курьера, где карты и когда будут, внести в CRM", r.actions)

    def test_far_offboarding_is_not_noise(self):
        case = {"id": "F", "type": "offboarding", "title": "t",
                "crm": {"last_day": "2026-10-26", "manager_confirmed": False}, "chat": []}
        self.assertEqual(s.assess(case, NOW, StubExtractor({})).level, s.OK)

    def test_onboarding_past_first_day_without_docs(self):
        case = {"id": "G", "type": "onboarding", "title": "t",
                "crm": {"first_day": "2026-10-16", "documents_received": False}, "chat": []}
        self.assertEqual(s.assess(case, NOW, StubExtractor({})).level, s.CRITICAL)


class TestFailSafe(unittest.TestCase):
    def test_broken_model_reply_keeps_case_visible(self):
        case = {"id": "H", "type": "onboarding", "title": "t",
                "crm": {"first_day": "2026-10-25", "documents_received": True},
                "chat": [{"author_role": "employee", "text": "кое-что поменялось"}]}
        r = s.assess(case, NOW, StubExtractor({"H": "Извините, я не могу помочь"}))
        self.assertEqual(r.level, s.CHECK)
        self.assertIn("Прочитать переписку по кейсу вручную", r.check)

    def test_validation_rejects_bad_output(self):
        bad = [
            {"messages": [{"index": 0, "claim": "resolved", "eta": None, "unofficial_channel": False}]},
            {"messages": [{"index": 0, "claim": "delivery_delayed", "eta": "после обеда", "unofficial_channel": False}]},
            {"messages": [{"index": 5, "claim": "other", "eta": None, "unofficial_channel": False}]},
            {"messages": []},
            "not json",
        ]
        for raw in bad:
            with self.assertRaises(s.ExtractionError):
                s.validate_extraction(raw, 1)

    def test_unknown_case_type_is_shown(self):
        case = {"id": "I", "type": "new_thing", "title": "t", "crm": {}, "chat": []}
        self.assertEqual(s.assess(case, NOW, StubExtractor({})).level, s.CHECK)


class TestPII(unittest.TestCase):
    def test_masking(self):
        text = "звоните +7 916 123-45-67, почта a.b@mail.ru, паспорт 45 12 345678, карта 2200 1234 5678 9012"
        masked = s.mask_pii(text)
        for leaked in ("916", "mail.ru", "345678", "9012"):
            self.assertNotIn(leaked, masked)

    def test_payload_to_llm_is_masked(self):
        case = {"type": "onboarding", "chat": [{"author_role": "employee", "text": "мой номер 89161234567"}]}
        self.assertNotIn("89161234567", json.dumps(s.build_user_payload(case), ensure_ascii=False))

    def test_telegram_digest_is_masked(self):
        case = {"id": "J", "type": "onboarding", "title": "t",
                "crm": {"first_day": "2026-10-16", "documents_received": False},
                "chat": [{"author_role": "employee", "text": "звоните +7 916 123-45-67"}]}
        r = s.assess(case, NOW, StubExtractor({"J": msg("other")}))
        self.assertNotIn("123-45-67", s.render_telegram([r], NOW))

    def test_times_are_not_masked(self):
        self.assertEqual(s.mask_pii("раньше 13:00 не приеду"), "раньше 13:00 не приеду")


if __name__ == "__main__":
    unittest.main()
