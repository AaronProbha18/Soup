"""#1192 — bundled mini_arithmetic must discriminate away from the 1.000 rail."""

from __future__ import annotations

from soup_cli.eval.forgetting import MINI_ARITHMETIC, _standalone_match, score_answer

# The multi-step rows that took the strong reference off the ceiling. Dropping
# any of them (or restoring the v0.73.2 single-step set) fails this file.
_HARD_ANSWERS = {
    "1786", "69104", "16", "75", "32768", "26", "259200", "998001",
    "31375", "120", "265", "90", "13717421", "360", "19/24", "20",
}


class TestArithmeticFixtureHasHeadroom:
    def test_fixture_keeps_40_rows(self):
        assert len(MINI_ARITHMETIC) == 40

    def test_the_last_16_rows_are_the_hard_set(self):
        assert {item["answer"] for item in MINI_ARITHMETIC[-16:]} == _HARD_ANSWERS

    def test_questions_are_unique(self):
        questions = [item["question"] for item in MINI_ARITHMETIC]
        assert len(questions) == len(set(questions))

    def test_no_hard_answer_is_echoed_by_its_question(self):
        # score_answer is a standalone-token match anywhere in the output, so a
        # model that repeats the question would score any answer it contains.
        for item in MINI_ARITHMETIC[-16:]:
            assert not _standalone_match(item["question"], item["answer"]), item

    def test_every_expected_answer_scores(self):
        for item in MINI_ARITHMETIC:
            assert score_answer(f"The answer is {item['answer']}.", item["answer"]), item


def test_fixture_change_bumps_baseline_provenance_revision():
    from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION

    assert BUNDLED_SCORER_REVISION >= 4
