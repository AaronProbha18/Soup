"""#1054: ``soup data preprocess`` cached rows with no ``labels`` column, so a
``pre_tokenized`` run trained on the prompt while the equivalent live run masked it.

The live SFT path builds ``labels`` through ``data/loss_mask.py``'s builders and
masks everything that is not an assistant response (``train_on_responses_only``,
default true). ``preprocess_dataset`` wrote only ``{input_ids, attention_mask}``,
so the cache reached TRL with no labels and TRL's collator fell back to
``labels = input_ids`` -- full-sequence loss. Two runs over the same rows therefore
optimised different objectives depending only on whether the data was preprocessed
first, and nothing raised: the loss curve looks entirely normal.

Because the failure is silent, every assertion here counts the **unmasked**
(``!= IGNORE_INDEX``) label positions. A present-but-degenerate ``labels`` array
(all ``-100``, or all unmasked) must fail these tests exactly like a missing one --
asserting merely that a ``labels`` column exists would pass the bug.

Three defects, three groups:
- the cache now carries a loss mask equal to the live path's (``TestCachedLabels``);
- the mask setting is a cache-key input, so a cache built under one masking config
  is refused by the other (``TestCacheKeyCoversMaskMode``);
- ``soup train`` reports the cached row count rather than 0 (``TestSampleCount``).

The tokenizer is a real ``transformers`` fast tokenizer built offline (same shape
as tests/test_issue791_preprocess_cache_eos.py), so ``apply_chat_template`` is the
genuine Jinja renderer.
"""

import json
from types import SimpleNamespace

import pytest

from soup_cli.data.loss_mask import IGNORE_INDEX

_SPECIALS = [
    "<unk>", "<s>", "</s>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
]
_WORDS = [
    "You", "are", "terse", ".", "What", "is", "the", "capital", "of", "France",
    "?", "Paris", "Berlin", "Germany", "system", "user", "assistant",
]
_BOS_ID = _SPECIALS.index("<s>")
_EOS_ID = _SPECIALS.index("</s>")

_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
)

_ROWS = [
    [
        {"role": "user", "content": "What is the capital of France ?"},
        {"role": "assistant", "content": "Paris ."},
    ],
    [
        {"role": "user", "content": "What is the capital of Germany ?"},
        {"role": "assistant", "content": "Berlin ."},
    ],
]


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, processors

    vocab = {token: index for index, token in enumerate(_SPECIALS + _WORDS)}
    backend = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(_SPECIALS)
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A",
        special_tokens=[("<s>", _BOS_ID), ("</s>", _EOS_ID)],
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = _TEMPLATE
    return tok


def _unmasked(labels):
    """Count of positions that contribute to the loss."""
    return sum(1 for label in labels if label != IGNORE_INDEX)


def _run_preprocess(tmp_path, monkeypatch, tok, *, rows, task="sft", data_extra=""):
    """Run the real ``soup data preprocess`` CLI and return the cache directory.

    Deliberately the CLI path, not a hand-built cache directory: the missing
    ``labels`` column was a defect of that code path specifically.
    """
    transformers = pytest.importorskip("transformers")
    from typer.testing import CliRunner

    from soup_cli.cli import app

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        f"base: x/y\ntask: {task}\n"
        "data:\n  train: ./d.jsonl\n  format: chatml\n"
        "  max_length: 128\n" + data_extra,
        encoding="utf-8",
    )
    (tmp_path / "d.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok
    )
    # Control the rows directly so the assertions are about masking, not format
    # conversion. preprocess_dataset local-imports this name.
    monkeypatch.setattr(
        "soup_cli.data.loader.load_dataset", lambda *a, **k: {"train": rows}
    )
    result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
    assert result.exit_code == 0, result.output
    cache_dirs = [p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()]
    assert len(cache_dirs) == 1, cache_dirs
    return cache_dirs[0]


def _load_cache(cache_dir):
    """Load the cache the way training loads it."""
    from soup_cli.utils.data_pipeline import load_pretokenized_dataset

    return load_pretokenized_dataset(str(cache_dir))


class TestCachedLabels:
    def test_cache_masks_the_same_tokens_the_live_path_masks(
        self, tmp_path, monkeypatch
    ):
        """Acceptance criterion #1. The cached rows' unmasked-token count equals
        the live path's, and is strictly greater than zero.

        ``build_assistant_only_labels`` IS the live ``train_on_responses_only``
        path (``data/sft_format.py`` calls it per row), so it is the baseline.
        The > 0 half matters on its own: a naive "the counts match" assertion
        passes when both sides degenerate to zero trained tokens -- the exact
        shape of the v0.73.3 all-``-100`` regression.

        regression: must fail without the fix -- pre-#1054 the cache has no
        ``labels`` column at all.
        """
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _tokenizer()
        rows = [{"messages": m} for m in _ROWS]
        ds = _load_cache(_run_preprocess(tmp_path, monkeypatch, tok, rows=rows))

        assert "labels" in ds.column_names, "the cache must carry a loss mask"
        assert len(ds) == len(_ROWS)
        for index, messages in enumerate(_ROWS):
            cached = list(ds[index]["labels"])
            live = build_assistant_only_labels(messages, tok, max_length=128)["labels"]
            assert _unmasked(live) > 0, "sanity: the live path trains on some tokens"
            assert _unmasked(cached) == _unmasked(live), (
                f"row {index}: cache trains on {_unmasked(cached)} tokens, "
                f"live path on {_unmasked(live)}"
            )
            assert len(cached) == len(ds[index]["input_ids"])

    def test_prompt_tokens_stay_masked(self, tmp_path, monkeypatch):
        """The defect's actual symptom: the prompt was trained on. The unmasked
        region must be a strict subset of the row, and must not cover the user
        turn's content tokens."""
        tok = _tokenizer()
        ds = _load_cache(
            _run_preprocess(
                tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
            )
        )
        labels = list(ds[0]["labels"])
        input_ids = list(ds[0]["input_ids"])
        assert 0 < _unmasked(labels) < len(labels), (
            "an all-masked or all-unmasked row is the silent failure this guards"
        )
        france_id = len(_SPECIALS) + _WORDS.index("France")
        assert france_id in input_ids, "guard: the prompt token is in the row"
        for token, label in zip(input_ids, labels):
            if token == france_id:
                assert label == IGNORE_INDEX, "prompt tokens must not be trained on"

    def test_input_ids_are_unchanged_by_the_labels_fix(self, tmp_path, monkeypatch):
        """Control: only ``labels`` is new. The cached ``input_ids`` keep the
        #785 BOS de-duplication and the #791 training EOS -- the byte-parity this
        path is deliberately pinned to."""
        tok = _tokenizer()
        ds = _load_cache(
            _run_preprocess(
                tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
            )
        )
        input_ids = list(ds[0]["input_ids"])
        assert input_ids.count(_BOS_ID) == 1, "#785: exactly one leading BOS"
        assert input_ids[-1] == _EOS_ID, "#791: the training EOS is still appended"

    def test_full_sequence_mode_trains_on_every_token(self, tmp_path, monkeypatch):
        """``train_on_responses_only: false`` caches an unmasked row -- the mask
        follows the config rather than being hardcoded."""
        tok = _tokenizer()
        ds = _load_cache(
            _run_preprocess(
                tmp_path,
                monkeypatch,
                tok,
                rows=[{"messages": _ROWS[0]}],
                data_extra="  train_on_responses_only: false\n",
            )
        )
        labels = list(ds[0]["labels"])
        assert labels == list(ds[0]["input_ids"])

    def test_pretrain_rows_are_unmasked(self, tmp_path, monkeypatch):
        """Control: pretraining trains on every token by design -- no masking is
        added to the raw-text branch."""
        tok = _tokenizer()
        ds = _load_cache(
            _run_preprocess(
                tmp_path,
                monkeypatch,
                tok,
                task="pretrain",
                rows=[{"text": "You are terse ."}],
            )
        )
        assert list(ds[0]["labels"]) == list(ds[0]["input_ids"])


class TestMissingLabelsIsRefused:
    def test_validate_raises_on_a_cache_without_labels(self):
        """A stale (pre-fix) or externally produced cache must be refused by name
        rather than silently skipping the causal-loss-target check -- skipping is
        what let the label-less cache reach TRL's ``labels = input_ids`` fallback.

        regression: must fail without the fix -- the old code returned here.
        """
        from soup_cli.trainer.sft import _validate_pretokenized_targets

        dataset = SimpleNamespace(column_names=["input_ids", "attention_mask"])
        with pytest.raises(ValueError, match="no 'labels' column"):
            _validate_pretokenized_targets(dataset, split="train", max_length=128)


class TestCacheKeyCoversMaskMode:
    def test_mask_mode_changes_the_key(self):
        """Acceptance criterion #2, at the hash. Same dataset / tokenizer /
        max_length / format under a different masking config must not collide."""
        from soup_cli.utils.data_pipeline import make_preprocess_cache_key

        common = dict(
            dataset_path="./d.jsonl",
            tokenizer_name="x/y",
            max_length=128,
            format_name="chatml",
        )
        keys = {
            make_preprocess_cache_key(**common, mask_mode=mode)
            for mode in ("responses_only", "train_field", "full")
        }
        assert len(keys) == 3, "each masking mode must hash to its own cache key"

    def test_train_on_eot_is_a_distinct_mask_mode(self):
        """``training.train_on_eot`` widens the live mask over the trailing EOT,
        so a cache built without it must not be reused by a run with it -- the
        same defect shape as #1054 itself, one config layer up."""
        from soup_cli.utils.data_pipeline import preprocess_mask_mode

        dcfg = SimpleNamespace(
            train_on_responses_only=True, train_on_messages_with_train_field=False
        )
        plain = preprocess_mask_mode(dcfg, SimpleNamespace(train_on_eot=False))
        eot = preprocess_mask_mode(dcfg, SimpleNamespace(train_on_eot=True))
        assert plain == "responses_only"
        assert eot == "responses_only+eot"
        assert plain != eot
        # train_on_eot only reaches the assistant-only builder, as in
        # build_format_row -- the other two modes ignore it.
        for other in (
            SimpleNamespace(
                train_on_responses_only=False,
                train_on_messages_with_train_field=True,
            ),
            SimpleNamespace(
                train_on_responses_only=False,
                train_on_messages_with_train_field=False,
            ),
        ):
            assert preprocess_mask_mode(
                other, SimpleNamespace(train_on_eot=True)
            ) == preprocess_mask_mode(other, SimpleNamespace(train_on_eot=False))

    def test_mask_mode_mirrors_build_format_row(self):
        """Both call sites derive the mode from one function, so they cannot
        drift the way the cache path drifted from the live path in #1054."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.utils.data_pipeline import preprocess_mask_mode

        assert preprocess_mask_mode(DataConfig(train="d.jsonl")) == "responses_only"
        assert (
            preprocess_mask_mode(
                DataConfig(train="d.jsonl", train_on_responses_only=False)
            )
            == "full"
        )
        assert (
            preprocess_mask_mode(
                DataConfig(
                    train="d.jsonl",
                    train_on_responses_only=False,
                    train_on_messages_with_train_field=True,
                )
            )
            == "train_field"
        )

    def test_cache_built_under_another_mask_mode_is_refused(
        self, tmp_path, monkeypatch
    ):
        """Acceptance criterion #2, end to end: a cache preprocessed with the
        default assistant-only mask is refused when loaded by a config that asks
        for the per-message ``train`` field mask.

        regression: must fail without the fix -- the key ignored the masking
        config, so the stale cache loaded silently under the wrong objective.
        """
        from rich.console import Console

        from soup_cli.config.schema import DataConfig
        from soup_cli.trainer.sft import _maybe_load_pretokenized

        tok = _tokenizer()
        cache_dir = _run_preprocess(
            tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
        )
        dcfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path=str(cache_dir.relative_to(tmp_path)),
            max_length=128,
            train_on_responses_only=False,
            train_on_messages_with_train_field=True,
        )
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _maybe_load_pretokenized(dcfg, "x/y", Console())

    def test_matching_mask_mode_still_loads(self, tmp_path, monkeypatch):
        """Control: the gate must not reject a cache built under the SAME config
        -- a key that rejects everything would pass the test above."""
        from rich.console import Console

        from soup_cli.config.schema import DataConfig
        from soup_cli.trainer.sft import _maybe_load_pretokenized

        tok = _tokenizer()
        cache_dir = _run_preprocess(
            tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
        )
        dcfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path=str(cache_dir.relative_to(tmp_path)),
            max_length=128,
        )
        loaded = _maybe_load_pretokenized(dcfg, "x/y", Console())
        assert loaded is not None
        train_ds, _ = loaded
        assert len(train_ds) == 1


class TestSampleCount:
    def test_pretokenized_count_comes_from_the_cache(self, tmp_path):
        """Acceptance criterion #3. ``load_dataset`` sees the ORIGINAL chatml file
        under a ``pre_tokenized`` config and drops every row for want of an
        ``input_ids`` column, so ``soup train`` printed "0 train samples" for a run
        that then trained on the whole cache.

        regression: must fail without the fix -- the count was
        ``len(dataset['train'])``, i.e. 0.
        """
        from soup_cli.commands.train import _train_sample_count

        cache = tmp_path / "cache"
        cache.mkdir(parents=True)
        (cache / "metadata.json").write_text(
            json.dumps({"cache_key": "deadbeef", "row_count": 6}), encoding="utf-8"
        )
        dcfg = SimpleNamespace(format="pre_tokenized", tokenized_path=str(cache))
        assert _train_sample_count(dcfg, {"train": []}) == 6

    def test_non_pretokenized_count_is_the_loaders_view(self):
        """Control: every other format still reports what the loader returned."""
        from soup_cli.commands.train import _train_sample_count

        dcfg = SimpleNamespace(format="chatml", tokenized_path=None)
        assert _train_sample_count(dcfg, {"train": [1, 2, 3]}) == 3

    def test_unreadable_metadata_falls_back(self, tmp_path):
        """Control: a cache without usable metadata must not crash the launch."""
        from soup_cli.commands.train import _train_sample_count

        dcfg = SimpleNamespace(
            format="pre_tokenized", tokenized_path=str(tmp_path / "missing")
        )
        assert _train_sample_count(dcfg, {"train": [1, 2]}) == 2
