"""#1192 — bundled mini_common_sense must discriminate away from the 1.000 rail."""

from __future__ import annotations

from collections import Counter

import pytest

from soup_cli.eval.forgetting import MINI_COMMON_SENSE

# Opening words of the 16 selected candidate rows, in fixture order. Dropping
# any of them (or restoring the v0.73.2 three-option set) fails this file.
_HARD_OPENINGS = (
    "You put a completely full, tightly sealed glass bottle",
    "A candle is burning inside a glass jar",
    "Your flight leaves at 7:00",
    "Which would NOT help a phone battery",
    "A heavy box and a light box",
    "A metal spoon and a wooden spoon",
    "Maria is taller than Ben, and Ben is taller than Chen",
    "Maria is taller than Ben, and Chen is taller than Ben",
    "You wash a wool sweater",
    "Ignoring air resistance",
    "Where would you see the stars",
    "It is 3 p.m. where you take off",
    "It is raining and the wind",
    "A fish tank has no lid",
    "You borrowed a friend's book",
    "A clock's hour hand points at 3",
)


class TestCommonSenseFixtureHasHeadroom:
    def test_fixture_keeps_40_rows(self):
        assert len(MINI_COMMON_SENSE) == 40

    def test_the_last_16_rows_are_the_selected_hard_set(self):
        hard = MINI_COMMON_SENSE[-16:]
        for item, opening in zip(hard, _HARD_OPENINGS, strict=True):
            assert item["question"].startswith(opening), item

    def test_hard_rows_offer_four_options(self):
        for item in MINI_COMMON_SENSE[-16:]:
            assert "(D)" in item["question"], item

    def test_questions_are_unique(self):
        questions = [item["question"] for item in MINI_COMMON_SENSE]
        assert len(questions) == len(set(questions))


class TestAnswerKeyIsNotLopsided:
    def test_no_letter_answers_more_than_30_percent(self):
        counts = Counter(item["answer"] for item in MINI_COMMON_SENSE)
        assert max(counts.values()) / len(MINI_COMMON_SENSE) <= 0.30, counts

    @pytest.mark.parametrize("letter", "ABCD")
    def test_a_constant_letter_model_cannot_score_above_30_percent(self, letter):
        from soup_cli.eval.gate_suites import score_bundled_suite

        assert score_bundled_suite("mini_common_sense", lambda _p: letter) <= 0.30


def test_fixture_change_bumps_baseline_provenance_revision():
    from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION

    assert BUNDLED_SCORER_REVISION >= 5
