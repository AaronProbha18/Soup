"""#1166 — three instrumentation gaps the gate-836 record found in ``soup bench train``.

1. The SM clock was one ``nvidia-smi`` read after training returned, on an idle
   card; it is now sampled during the counted steps.
2. ``timing`` carried only median and p95; it now carries every counted step.
3. The docs said the report is written for every failure. Pre-flight refusals
   write nothing; only post-run checks leave a ``valid: false`` report.
"""

from __future__ import annotations

import json

from soup_cli.bench.collector import BenchCollector
from soup_cli.bench.train_report import summarize_clock_samples, summarize_step_times
from tests import test_issue836_bench_train_cli as cli_tests
from tests.conftest import strip_ansi

workdir = cli_tests.workdir
runner = cli_tests.runner


class TestTheClockIsReadWhileBusy:
    # Warm-up ramps at t<10, the counted steps run 10..20, the card idles after.
    SAMPLES = [(1.0, 300), (5.0, 1400), (10.0, 2400), (12.0, 2550), (15.0, 2370),
               (20.0, 2500), (21.0, 285), (25.0, 180)]

    def test_only_the_counted_window_is_summarised(self):
        summary = summarize_clock_samples(self.SAMPLES, 10.0, 20.0)
        assert summary == {
            "min": 2370, "median": 2450.0, "max": 2550, "sample_count": 4,
            "unavailable_reason": None,
        }

    def test_a_busy_reading_is_not_the_idle_one(self):
        """The old field was the last sample; the gate record measured it at
        180-285 MHz against a busy median of 2370-2557."""
        idle_after_run = self.SAMPLES[-1][1]
        busy = summarize_clock_samples(self.SAMPLES, 10.0, 20.0)
        assert busy["min"] > idle_after_run * 5

    def test_no_sample_in_the_window_is_none_not_a_guess(self):
        summary = summarize_clock_samples(self.SAMPLES, 100.0, 200.0)
        assert (summary["min"], summary["median"], summary["max"]) == (None, None, None)
        assert summary["sample_count"] == 0
        assert summarize_clock_samples(self.SAMPLES, None)["sample_count"] == 0

    def test_no_tool_says_so(self):
        summary = summarize_clock_samples([], None, unavailable_reason="no nvidia-smi tool found")
        assert summary["unavailable_reason"] == "no nvidia-smi tool found"

    def test_the_collector_marks_the_counted_window(self):
        clock = iter(float(i) for i in range(10))
        collector = BenchCollector(warmup_steps=2)
        collector._now = lambda: next(clock)
        for _ in range(4):
            collector.on_step_begin(None, None, None)
            collector.on_step_end(None, None, None)
        # Steps end at 1, 2, 3, 4: warm-up is 0..2, the counted window 2..4.
        assert (collector.counted_window_started, collector.counted_window_ended) == (2.0, 4.0)

    def test_the_sampler_reads_the_clock_through_the_resolved_tool(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.bench.train_run import ClockSampler

        script = cli_tests.TestDriverAndClockProvenance._fake_tool(
            tmp_path, monkeypatch, "2505\n"
        )
        # stop() waits for the tick in flight, so at least one always lands.
        samples = ClockSampler(interval=0.01).start().stop()
        assert samples and {clock for _, clock in samples} == {2505}
        asked = (tmp_path / (script.name + ".args")).read_text(encoding="utf-8")
        assert "clocks.sm" in asked and "memory" not in asked

    def test_an_unreadable_clock_is_skipped(self, tmp_path, monkeypatch):
        from soup_cli.bench.train_run import ClockSampler

        cli_tests.TestDriverAndClockProvenance._fake_tool(tmp_path, monkeypatch, "[N/A]\n")
        assert ClockSampler(interval=0.01).start().stop() == []

    def test_without_a_tool_nothing_starts(self, monkeypatch):
        from soup_cli.bench.train_run import ClockSampler

        monkeypatch.setattr("soup_cli.utils.layer_stream._resolve_tool", lambda *_a: None)
        sampler = ClockSampler()
        assert sampler.start().stop() == [] and sampler.tool is None


class TestEveryCountedStepIsKept:
    def test_step_seconds_are_the_counted_steps_in_run_order(self):
        """A mode change at step 5: median and p95 show one happened, the
        sequence shows where."""
        times = [9.0, 9.0] + [1.0] * 5 + [2.0] * 5
        summary = summarize_step_times(times, warmup_steps=2)
        assert summary["step_seconds"] == [1.0] * 5 + [2.0] * 5
        assert len(summary["step_seconds"]) == summary["counted_steps"]


class TestAPreFlightRefusalWritesNoReport:
    def test_zero_trainable_parameters(self, workdir, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        setup = SFTTrainerWrapper.setup

        def setup_then_freeze(self, dataset):
            setup(self, dataset)
            for param in self.trainer.model.parameters():
                param.requires_grad_(False)

        monkeypatch.setattr(SFTTrainerWrapper, "setup", setup_then_freeze)
        result = runner.invoke(
            cli_tests.app, ["bench", "train", "--config", "soup.yaml", "--steps", "3",
                            "--warmup", "1", "-o", "r.json"],
        )
        assert result.exit_code == 1
        assert "0 trainable parameter tensors" in strip_ansi(result.output)
        assert not (workdir / "r.json").exists()

    def test_warmup_that_eats_every_step(self, workdir):
        result = runner.invoke(
            cli_tests.app, ["bench", "train", "--config", "soup.yaml", "--steps", "2",
                            "--warmup", "2", "-o", "r.json"],
        )
        assert result.exit_code == 1
        assert not (workdir / "r.json").exists()

    def test_a_post_run_failure_does_write_one(self, workdir):
        """The contrast: a check that fires after training keeps its evidence."""
        # Exercised in full by test_a_run_that_does_not_train_exits_non_zero_and_still_writes;
        # here only the shape a parser of the file depends on.
        report = cli_tests._run(steps=2, warmup=1)(
            before_train=lambda w: setattr(w.trainer.args, "learning_rate", 0.0)
        )
        assert report["valid"] is False
        json.dumps(report, default=str)
        assert report["provenance"]["sm_clock_mhz_busy"]["unavailable_reason"] == (
            "not a CUDA run"
        )
