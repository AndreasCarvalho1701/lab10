import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

import lab10_pipeline
from lab10_attention import AttentionChoice


class FakeConfig:
    use_cache = True


class FakeModel:
    def __init__(self):
        self.config = FakeConfig()

    def generate(self, input_ids, max_new_tokens, **_kwargs):
        extra = torch.ones((input_ids.shape[0], max_new_tokens), dtype=input_ids.dtype)
        return torch.cat([input_ids, extra], dim=-1)


class FakeTokenizer:
    eos_token_id = 0


class PipelineCoreTests(unittest.TestCase):
    def test_fake_context_has_expected_size(self):
        context = lab10_pipeline.build_fake_medical_context(12000)
        self.assertGreaterEqual(len(context.split()), 10000)
        self.assertIn("diabetes mellitus tipo 2", context)

    def test_generate_once_respects_cache_flag_and_token_count(self):
        model = FakeModel()
        input_ids = torch.ones((1, 8), dtype=torch.long)

        metrics = lab10_pipeline.generate_once(
            model=model,
            tokenizer=FakeTokenizer(),
            input_ids=input_ids,
            max_new_tokens=100,
            use_cache=False,
        )

        self.assertFalse(model.config.use_cache)
        self.assertEqual(metrics["generated_tokens"], 100)
        self.assertEqual(metrics["use_cache"], False)
        self.assertIn("seconds", metrics)
        self.assertIn("peak_vram_mb", metrics)

    def test_run_writes_expected_metrics_shape_with_mocks(self):
        class RunTokenizer:
            eos_token_id = 0

            def __call__(self, prompt, return_tensors, truncation):
                self.prompt = prompt
                return {"input_ids": torch.ones((1, 16), dtype=torch.long)}

        class RunModel(FakeModel):
            device = "cpu"

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "metrics.json"
            args = Namespace(
                model_id="modelo-falso",
                target_context_tokens=12000,
                max_new_tokens=100,
                disable_flash=True,
                output=output,
            )

            with patch("lab10_pipeline.detect_attention_implementation") as detect, patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=RunTokenizer()
            ), patch("lab10_pipeline.load_quantized_model", return_value=RunModel()):
                detect.return_value = AttentionChoice(
                    implementation="sdpa",
                    reason="teste",
                    flash_available=False,
                )
                metrics = lab10_pipeline.run(args)
                self.assertTrue(output.exists())

        self.assertEqual(metrics["attention_implementation"], "sdpa")
        self.assertEqual(metrics["no_cache"]["generated_tokens"], 100)
        self.assertEqual(metrics["with_cache"]["generated_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
