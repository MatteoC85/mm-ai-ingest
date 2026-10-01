"""Offline regressions using the real Smart endpoints and simulated model outputs.

Network and database connections are denied before importing the application.
These checks verify contracts and grounding rules, not live model quality.
"""
import copy
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_TEST_EXTERNAL_IO_DENIED")


_socket_connect = socket.socket.connect


def local_event_loop_connect(sock, address):
    # Windows implements asyncio's private socketpair over loopback TCP.
    if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
        return _socket_connect(sock, address)
    return denied()


class SmartDiagnosticQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {
            "MM_USAGE_ENFORCEMENT": "off", "AI_INTERNAL_SECRET": "offline-smart-fixture-only",
            "OPENAI_API_KEY": "offline-unused", "MM_INGEST_LEDGER_AUTO_DDL": "0",
        }))
        cls.guards.enter_context(patch.object(socket.socket, "connect", local_event_loop_connect))
        cls.guards.enter_context(patch.object(socket, "create_connection", denied))
        cls.guards.enter_context(patch("requests.sessions.Session.request", denied))
        cls.guards.enter_context(patch("psycopg2.connect", denied))
        cls.m = importlib.import_module("main")
        from fastapi.testclient import TestClient
        cls.client = TestClient(cls.m.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.guards.close()

    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(self.m, "AI_INTERNAL_SECRET", "offline-smart-fixture-only"))
        self.patches.enter_context(patch.object(self.m, "SMART_DIAGNOSTIC_ENABLED", True))
        self.patches.enter_context(patch.object(self.m, "ASSISTANT_CORE_V2_ENABLED", True))
        self.patches.enter_context(patch.object(self.m, "_sd_enrich_state_evidence_from_answer", side_effect=lambda **kw: kw["state"]))
        self.patches.enter_context(patch.object(self.m, "_build_rg_links", return_value=[]))
        self.hyps = [
            {"id": "H1", "rank": 1, "label": "Supply pressure loss", "description": "Pressure condition", "why": "The manual links low pressure to failed motion.", "probability_pct": 60, "probability_band": "medium", "status": "open", "checks": ["Read the pressure indicator."], "evidence_ids": ["manual:p7"]},
            {"id": "H2", "rank": 2, "label": "Position signal absent", "description": "Signal condition", "why": "The fault chart requires position confirmation.", "probability_pct": 40, "probability_band": "medium", "status": "open", "checks": ["Observe the position input on the HMI."], "evidence_ids": ["manual:p8"]},
        ]
        self.question = {"question_id": "Q1", "question_number": 1, "question_type": "yes_no", "question_text": "Is the pressure indicator within the documented range?", "why_asked": "Separate supply from position faults.", "safety_level": "normal", "safety_note": "Observe from the operator panel.", "options": self.m._sd_default_yes_no_options("en"), "target_hypotheses": ["H1", "H2"]}
        self.state = {"company_id": "fixture-company", "machine_id": "fixture-machine", "session_id": "fixture-session", "status": "in_progress", "language": "en", "symptom_text": "Actuator will not move", "max_questions": 6, "max_hypotheses": 4, "history": [], "current_question": self.question, "current_step_number": 1, "hypotheses": self.hyps, "evidence_gate": {"accepted": True, "relevant_evidence_ids": ["manual:p7", "manual:p8"]}, "evidence": [
            {"citation_id": "manual:p7", "bubble_document_id": "manual", "source_type": "document", "display_title": "Operator manual", "display_label": "Operator manual - p. 7", "page_from": 7, "page_to": 7, "snippet": "Low supply pressure prevents motion."},
            {"citation_id": "manual:p8", "bubble_document_id": "manual", "source_type": "document", "display_title": "Operator manual", "display_label": "Operator manual - p. 8", "page_from": 8, "page_to": 8, "snippet": "Position confirmation is required."},
        ]}

    def parsed_step(self, **kw):
        q = {**self.question, "question_id": "Q2", "question_number": 2, "question_text": "Does the HMI show position confirmation?"}
        return {"status": "in_progress", "final_ready": False, "operator_summary": "Pressure checked; inspect the position signal.", "question": q, "hypotheses": copy.deepcopy(self.hyps), "final_result": {}, **kw}

    def post(self, kind, *, state=None, answer=None, **kw):
        data = {"company_id": "fixture-company", "machine_id": "fixture-machine", "session_id": "fixture-session", "language": "en", "state_json": self.m._sd_sign_state(copy.deepcopy(state or self.state))}
        if kind == "answer":
            data.update(question_id="Q1", answer=answer or {"value": "yes", "api_value": "yes", "label": "Yes", "free_text": ""})
        data.update(kw)
        return self.client.post("/v1/ai/smart-diagnostic/" + kind, json=data, headers={"x-ai-internal-secret": "offline-smart-fixture-only"})

    def test_finalizer_cannot_select_excluded_hypothesis(self):
        self.hyps[0]["status"] = "excluded"
        with patch.object(self.m, "_sd_llm_finalize", return_value={"most_likely_hypothesis_id": "H1"}):
            response = self.post("finalize")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["final_result"]["most_likely_hypothesis_id"], "H2")
        self.assertEqual(body["operator_summary"], body["final_summary_text"])

    def test_answer_final_cannot_select_excluded_or_invent_summary_checks(self):
        hyps = copy.deepcopy(self.hyps)
        hyps[0]["status"] = "excluded"
        parsed = self.parsed_step(status="completed", final_ready=True, hypotheses=hyps, operator_summary="Replace an undocumented controller; certainty 100%.", final_result={"most_likely_hypothesis_id": "H1", "recommended_checks": ["Read the pressure indicator.", "Replace controller."]})
        with patch.object(self.m, "_sd_llm_step_answer", return_value=parsed):
            response = self.post("answer")
        body = response.json()
        self.assertEqual(body["final_result"]["most_likely_hypothesis_id"], "H2")
        self.assertEqual(body["operator_summary"], body["final_summary_text"])
        self.assertEqual(body["final_result"]["recommended_checks"], ["Observe the position input on the HMI"])

    def test_all_excluded_does_not_manufacture_a_final_diagnosis(self):
        for h in self.hyps:
            h["status"] = "excluded"
        with patch.object(self.m, "_sd_llm_finalize", return_value={"most_likely_hypothesis_id": "H1"}) as provider:
            response = self.post("finalize")
        self.assertFalse(response.json()["final_ready"])
        self.assertEqual(response.json()["status"], "no_sources")
        provider.assert_not_called()

    def test_question_limit_is_enforced_when_model_keeps_asking(self):
        self.state["max_questions"] = 1
        with patch.object(self.m, "_sd_llm_step_answer", return_value=self.parsed_step()):
            response = self.post("answer")
        body = response.json()
        self.assertTrue(body["final_ready"])
        self.assertEqual(body["status"], "completed")
        self.assertFalse(body["question_text"])
        self.assertEqual(len(json.loads(body["session_state_json"])["history"]), 1)

    def test_question_number_is_server_monotonic(self):
        parsed = self.parsed_step()
        parsed["question"]["question_number"] = 1
        with patch.object(self.m, "_sd_llm_step_answer", return_value=parsed):
            response = self.post("answer")
        self.assertEqual(response.json()["question_number"], 2)

    def test_closed_answer_uses_one_canonical_value_and_label(self):
        for language, question_type, selected_id in (
            ("en", "yes_no", "yes"),
            ("it", "yes_no", "unknown"),
            ("en", "single_choice", "warm"),
        ):
            with self.subTest(language=language, question_type=question_type, selected_id=selected_id):
                state = copy.deepcopy(self.state)
                if question_type == "single_choice":
                    state["current_question"].update(question_type=question_type, options=[
                        {"id": "cold", "label_it": "A freddo", "label_en": "Cold"},
                        {"id": "warm", "label_it": "A caldo", "label_en": "Warm"},
                    ])
                selected = next(option for option in state["current_question"]["options"] if option["id"] == selected_id)
                observation = "The indicator is unreadable; the operator has not confirmed its value."
                with patch.object(self.m, "_sd_llm_step_answer", return_value=self.parsed_step()) as provider:
                    response = self.post("answer", state=state, language=language, answer={
                        "value": "opposite of the selected option", "api_value": selected_id,
                        "label": "client label contradicting the selected option", "free_text": observation,
                    })
                self.assertEqual(response.status_code, 200, response.text)
                returned = json.loads(response.json()["session_state_json"])
                for answer in (provider.call_args.kwargs["answer"], returned["history"][-1]["answer"]):
                    self.assertEqual(answer["value"], selected_id)
                    self.assertEqual(answer["api_value"], selected_id)
                    self.assertEqual(answer["label"], selected[f"label_{language}"])
                    self.assertEqual(answer["free_text"], observation)

    def test_terminal_step_counts_answered_questions_without_exceeding_limit(self):
        for maximum, answered, model_completed in ((1, 1, False), (8, 8, False), (6, 1, True)):
            with self.subTest(maximum=maximum, answered=answered, model_completed=model_completed):
                state = copy.deepcopy(self.state)
                state["max_questions"] = maximum
                state["history"] = [
                    {"question": {**self.question, "question_id": f"Q{n}", "question_number": n,
                        "question_text": f"Previously answered question {n}?"},
                     "answer": {"value": "unknown", "api_value": "unknown", "label": "Unknown", "free_text": ""}}
                    for n in range(1, answered)
                ]
                state["current_question"].update(question_id=f"Q{answered}", question_number=answered)
                state["current_step_number"] = answered
                parsed = self.parsed_step()
                if model_completed:
                    parsed.update(status="completed", final_ready=True)
                with patch.object(self.m, "_sd_llm_step_answer", return_value=parsed):
                    response = self.post("answer", state=state, question_id=f"Q{answered}")
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                returned = json.loads(body["session_state_json"])
                self.assertTrue(body["final_ready"])
                self.assertEqual(body["question_number"], answered)
                self.assertEqual(body["question"]["question_number"], answered)
                self.assertEqual(returned["current_step_number"], answered)
                self.assertLessEqual(returned["current_step_number"], maximum)
                self.assertEqual(len(returned["history"]), answered)
                self.assertEqual(returned["current_question"], {})
                self.assertEqual(returned["state_signature"], self.m._sd_state_signature(returned))

    def test_unknown_option_is_rejected_before_provider(self):
        with patch.object(self.m, "_sd_llm_step_answer", return_value=self.parsed_step()) as provider:
            response = self.post("answer", answer={"value": "invented", "api_value": "invented"})
        self.assertEqual(response.status_code, 400)
        provider.assert_not_called()

    def test_completed_state_rejects_answer_before_provider(self):
        self.state.update(status="completed", current_question={})
        with patch.object(self.m, "_sd_llm_step_answer", return_value=self.parsed_step()) as provider:
            response = self.post("answer")
        self.assertEqual(response.status_code, 409)
        provider.assert_not_called()

    def test_yes_no_unknown_and_free_text_reach_model_without_positive_coercion(self):
        with patch.object(self.m, "_sd_llm_step_answer", return_value=self.parsed_step()) as provider:
            response = self.post("answer", answer={"value": "unknown", "api_value": "unknown", "label": "I don't know", "free_text": "Indicator is unreadable."})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(provider.call_args.kwargs["answer"]["api_value"], "unknown")
        self.assertEqual(provider.call_args.kwargs["answer"]["free_text"], "Indicator is unreadable.")

    def test_single_choice_preserves_bubble_contract_and_history(self):
        self.question.update(question_type="single_choice", options=[{"id": "cold", "label_it": "A freddo", "label_en": "Cold"}, {"id": "warm", "label_it": "A caldo", "label_en": "Warm"}])
        with patch.object(self.m, "_sd_llm_step_answer", return_value=self.parsed_step()) as provider:
            response = self.post("answer", answer={"value": "warm", "api_value": "warm", "label": "Warm", "free_text": "After two hours."})
        body = response.json()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(provider.call_args.kwargs["state"]["history"][-1]["answer"]["api_value"], "warm")
        self.assertEqual(body["question"]["question_id"], body["question_id"])
        self.assertEqual(json.loads(body["hypotheses_json"]), body["hypotheses"])
        self.assertIn("c1_citation_id", body)
        self.assertIn("final_recommended_checks_json", body)

    def test_repeated_question_is_not_published_as_progress(self):
        parsed = self.parsed_step(question={**self.question, "question_text": "  IS THE PRESSURE INDICATOR WITHIN THE DOCUMENTED RANGE??? "})
        with patch.object(self.m, "_sd_llm_step_answer", return_value=parsed) as provider:
            response = self.post("answer")
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["detail"]["code"], "SMART_DIAGNOSTIC_REPEATED_QUESTION")
        self.assertEqual(provider.call_count, 1)

    def test_scope_mismatch_and_tamper_are_rejected_before_provider(self):
        for case in ("company", "machine", "tampered"):
            with self.subTest(case=case):
                data = {"company_id": "fixture-company", "machine_id": "fixture-machine", "session_id": "fixture-session", "question_id": "Q1", "answer": {"value": "yes"}, "state_json": self.m._sd_sign_state(copy.deepcopy(self.state))}
                if case == "company":
                    data["company_id"] = "other-company"
                elif case == "machine":
                    data["machine_id"] = "other-machine"
                else:
                    data["state_json"]["hypotheses"][0]["probability_pct"] = 100
                with patch.object(self.m, "_sd_llm_step_answer") as provider:
                    response = self.client.post("/v1/ai/smart-diagnostic/answer", json=data, headers={"x-ai-internal-secret": "offline-smart-fixture-only"})
                self.assertEqual(response.status_code, 400, response.text)
                provider.assert_not_called()

    def test_clearing_signature_cannot_bypass_state_integrity(self):
        for endpoint in ("answer", "finalize"):
            with self.subTest(endpoint=endpoint):
                state = self.m._sd_sign_state(copy.deepcopy(self.state))
                state.pop("state_signature")
                state["hypotheses"][0]["probability_pct"] = 100
                data = {"company_id": "fixture-company", "machine_id": "fixture-machine", "session_id": "fixture-session", "question_id": "Q1", "answer": {"value": "yes"}, "state_json": state}
                with patch.object(self.m, "_sd_llm_step_answer", return_value=self.parsed_step()) as answer_provider, patch.object(self.m, "_sd_llm_finalize", return_value={"most_likely_hypothesis_id": "H1"}) as final_provider:
                    response = self.client.post("/v1/ai/smart-diagnostic/" + endpoint, json=data, headers={"x-ai-internal-secret": "offline-smart-fixture-only"})
                self.assertEqual(response.status_code, 400, response.text)
                answer_provider.assert_not_called()
                final_provider.assert_not_called()

    def test_returned_signed_state_continues_opaquely_without_resigning(self):
        self.state["max_questions"] = 2
        with patch.object(self.m, "_sd_llm_step_answer", return_value=self.parsed_step()):
            first = self.post("answer")
        self.assertEqual(first.status_code, 200)
        next_turn = first.json()
        data = {"company_id": "fixture-company", "machine_id": "fixture-machine", "session_id": "fixture-session", "question_id": next_turn["question_id"], "answer": {"value": "no", "api_value": "no", "label": "No", "free_text": "Position is not indicated."}, "language": "en", "state_json": next_turn["session_state_json"]}
        parsed = self.parsed_step()
        parsed["hypotheses"][0]["status"] = "excluded"
        with patch.object(self.m, "_sd_llm_step_answer", return_value=parsed):
            second = self.client.post("/v1/ai/smart-diagnostic/answer", json=data, headers={"x-ai-internal-secret": "offline-smart-fixture-only"})
        self.assertEqual(second.status_code, 200, second.text)
        body = second.json()
        self.assertTrue(body["final_ready"])
        self.assertEqual(body["final_result"]["most_likely_hypothesis_id"], "H2")
        self.assertEqual(len(json.loads(body["session_state_json"])["history"]), 2)

    def test_model_cannot_add_unadmitted_hypothesis_or_question_target(self):
        parsed = self.parsed_step()
        parsed["hypotheses"].append({**self.hyps[0], "id": "H9", "label": "Invented cause", "evidence_ids": ["foreign-document:p1"]})
        parsed["question"]["target_hypotheses"] = ["H9"]
        with patch.object(self.m, "_sd_llm_step_answer", return_value=parsed):
            response = self.post("answer")
        body = response.json()
        self.assertEqual({h["id"] for h in body["hypotheses"]}, {"H1", "H2"})
        self.assertTrue(set(body["question"]["target_hypotheses"]).issubset({"H1", "H2"}))
        self.assertNotIn("foreign-document", body["session_state_json"])

    def test_final_sources_and_bubble_flattened_fields_remain_consistent(self):
        self.state["citations"] = [{**e, "snippet_clean": e["snippet"]} for e in self.state["evidence"]]
        with patch.object(self.m, "_sd_llm_finalize", return_value={"most_likely_hypothesis_id": "H2", "most_likely_label": "Invented", "probability_pct": 100, "recommended_checks": ["Invented operation"]}):
            response = self.post("finalize")
        body = response.json()
        self.assertEqual(body["final_most_likely_label"], "Position signal absent")
        self.assertEqual(body["final_probability_pct"], 40)
        self.assertEqual(json.loads(body["final_result_json"]), body["final_result"])
        self.assertEqual(json.loads(body["citations_json"]), body["citations"])
        self.assertIn("manual:p8", {c["citation_id"] for c in body["citations"]})
        self.assertTrue(all(c["bubble_document_id"] == "manual" for c in body["citations"]))
        self.assertEqual(body["c1_citation_id"], body["citations"][0]["citation_id"])
        state = json.loads(body["session_state_json"])
        self.assertEqual(state["state_signature"], self.m._sd_state_signature(state))
        self.assertEqual(state["current_question"], {})

    def test_start_core_route_preserves_valid_guided_turn(self):
        step = self.m._sd_normalize_step(self.parsed_step(), language="en", question_number_default=1, max_hypotheses=4, allowed_evidence_ids={"manual:p7", "manual:p8"})
        final = self.m._sd_response_from_step(session_id="fixture-session", company_id="fixture-company", machine_id="fixture-machine", symptom_text="Actuator will not move", language="en", state=self.state, step=step, citations=self.state["evidence"], rg_links=[])
        final.update(effective_mode="smart_diagnostic")
        with patch.object(self.m._ASSISTANT_CORE_SMART_ENGINE, "run", return_value=final):
            response = self.post("start", symptom_text="Actuator will not move")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["question_number"], 1)
        self.assertEqual(body["question_id"], "Q1")
        self.assertEqual(body["option_count"], 3)
        self.assertFalse(body["final_ready"])

    def test_start_cannot_publish_an_ask_answer_as_guided_session(self):
        with patch.object(self.m._ASSISTANT_CORE_SMART_ENGINE, "run", return_value={"effective_mode": "ask", "status": "answered", "answer": "Generic answer"}):
            response = self.post("start", symptom_text="Actuator will not move")
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["detail"]["code"], "SMART_DIAGNOSTIC_MODE_CONTINUITY_FAILED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
