"""
Training callback for periodic steganography evaluation.

This callback can be integrated into the HuggingFace Trainer to:
- Periodically evaluate generation alignment during training
- Optionally evaluate detection capability
- Log metrics to wandb
- Save detailed results to JSONL files

Usage:
    from eval_callback import StegEvalCallback

    callback = StegEvalCallback(
        tokenizer=tokenizer,
        eval_every_n_steps=100,
        num_samples=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[callback],
        ...
    )
"""
import gc
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import wandb
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from evaluation import (
    StegEvaluator,
    load_eval_prompts,
    get_wandb_metrics,
    results_to_dict,
    EvaluationResults,
)


class StegEvalCallback(TrainerCallback):
    """
    Training callback for periodic steganography evaluation.

    Evaluates model's watermark generation capability at regular intervals
    and logs metrics to wandb.
    """

    def __init__(
        self,
        tokenizer,
        eval_every_n_steps: int = 100,
        num_samples: int = 5,
        max_new_tokens: int = 256,
        prompts_dir: Optional[Path] = None,
        logs_dir: Optional[str] = None,
        run_detection: bool = False,
        randomize_prompts: bool = True,
        log_samples_to_wandb: bool = True,
    ):
        """
        Initialize the evaluation callback.

        Args:
            tokenizer: The tokenizer (must match the training tokenizer)
            eval_every_n_steps: Run evaluation every N training steps
            num_samples: Number of generation samples per mode (red/blue)
            max_new_tokens: Maximum tokens to generate per sample
            prompts_dir: Directory containing prompt templates
            logs_dir: Directory to save JSONL log files
            run_detection: Whether to also run detection evaluation
            randomize_prompts: Randomly select prompts each evaluation
            log_samples_to_wandb: Log sample generations as wandb tables
        """
        self.tokenizer = tokenizer
        self.eval_every_n_steps = eval_every_n_steps
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        self.run_detection = run_detection
        self.randomize_prompts = randomize_prompts
        self.log_samples_to_wandb = log_samples_to_wandb

        # Load all available prompts
        self.all_prompts = load_eval_prompts(self.prompts_dir)

        # Setup logging directory
        if logs_dir:
            self.logs_dir = Path(logs_dir)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = self.logs_dir / f"steg_eval_{timestamp}.jsonl"
        else:
            self.logs_dir = None
            self.log_file = None

        # Track evaluation history for plotting
        self.eval_history = []

    def _get_prompts(self) -> List[str]:
        """Get prompts for this evaluation round."""
        if self.randomize_prompts and len(self.all_prompts) > self.num_samples:
            return random.sample(self.all_prompts, self.num_samples)
        return self.all_prompts[:self.num_samples]

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Run evaluation at specified intervals."""
        if state.global_step == 0:
            return

        if state.global_step % self.eval_every_n_steps != 0:
            return

        if model is None:
            return

        self._run_evaluation(model, state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Run final evaluation at end of training."""
        if model is not None:
            self._run_evaluation(model, state.global_step, prefix="final_")

    def _run_evaluation(
        self,
        model,
        step: int,
        prefix: str = "",
    ):
        """
        Run steganography evaluation and log results.

        Args:
            model: The model to evaluate
            step: Current training step
            prefix: Prefix for metric names (e.g., "final_")
        """
        print(f"\n{'='*50}")
        print(f"Steganography Evaluation (Step {step})")
        print(f"{'='*50}")

        model.eval()

        # Create evaluator
        evaluator = StegEvaluator(
            model=model,
            tokenizer=self.tokenizer,
            prompts_dir=self.prompts_dir,
        )

        # Get prompts for this round
        prompts = self._get_prompts()

        # Run generation evaluation
        try:
            gen_results = evaluator.evaluate_generation(
                prompts=prompts,
                num_samples=len(prompts),
                max_new_tokens=self.max_new_tokens,
            )
        except Exception as e:
            print(f"Error during generation evaluation: {e}")
            gen_results = []

        # Create results object
        results = EvaluationResults(timestamp=datetime.now().isoformat())
        results.generation_results = gen_results
        results.num_generation_samples = len(gen_results)

        # Compute generation metrics
        red_results = [r for r in gen_results if r.mode == "red"]
        blue_results = [r for r in gen_results if r.mode == "blue"]

        if red_results:
            results.red_alignment_avg = sum(r.alignment for r in red_results) / len(red_results)
            results.red_watermark_rate = sum(r.is_watermarked for r in red_results) / len(red_results)

        if blue_results:
            results.blue_alignment_avg = sum(r.alignment for r in blue_results) / len(blue_results)
            results.blue_watermark_rate = sum(r.is_watermarked for r in blue_results) / len(blue_results)

        if gen_results:
            results.overall_alignment_avg = (results.red_alignment_avg + results.blue_alignment_avg) / 2
            results.overall_watermark_rate = (results.red_watermark_rate + results.blue_watermark_rate) / 2

        # Optionally run detection evaluation
        if self.run_detection and gen_results:
            try:
                test_samples = [
                    {"text": r.generated_text, "mode": r.mode, "parity": r.parity}
                    for r in gen_results if len(r.generated_text) > 50
                ]
                if test_samples:
                    det_results = evaluator.evaluate_detection(test_samples)
                    results.detection_results = det_results
                    results.num_detection_samples = len(test_samples)

                    red_first = [r for r in det_results if r.prompt_variant == "red_first"]
                    blue_first = [r for r in det_results if r.prompt_variant == "blue_first"]

                    if red_first:
                        results.detection_accuracy_red_first = sum(r.correct for r in red_first) / len(red_first)
                    if blue_first:
                        results.detection_accuracy_blue_first = sum(r.correct for r in blue_first) / len(blue_first)

                    results.detection_accuracy_avg = (
                        results.detection_accuracy_red_first + results.detection_accuracy_blue_first
                    ) / 2
            except Exception as e:
                print(f"Error during detection evaluation: {e}")

        # Print summary
        print(f"Red alignment:     {results.red_alignment_avg:.1%}")
        print(f"Blue alignment:    {results.blue_alignment_avg:.1%}")
        print(f"Average alignment: {results.overall_alignment_avg:.1%}")
        if self.run_detection:
            print(f"Detection accuracy: {results.detection_accuracy_avg:.1%}")
        print(f"{'='*50}\n")

        # Log to wandb
        self._log_to_wandb(results, step, prefix)

        # Save to JSONL file
        self._save_to_jsonl(results, step, prefix)

        # Track history
        self.eval_history.append({
            "step": step,
            "red_alignment": results.red_alignment_avg,
            "blue_alignment": results.blue_alignment_avg,
            "avg_alignment": results.overall_alignment_avg,
        })

        # Cleanup
        del evaluator, gen_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()

    def _log_to_wandb(
        self,
        results: EvaluationResults,
        step: int,
        prefix: str = "",
    ):
        """Log metrics to wandb."""
        if wandb.run is None:
            return

        # Log scalar metrics
        metrics = get_wandb_metrics(results, prefix)
        wandb.log(metrics, step=step)

        # Log sample generations as a table
        if self.log_samples_to_wandb and results.generation_results:
            columns = ["mode", "prompt", "alignment", "watermarked", "text_preview"]
            table = wandb.Table(columns=columns)

            for r in results.generation_results[:10]:  # Limit to 10 samples
                table.add_data(
                    r.mode,
                    r.prompt[:80],
                    f"{r.alignment:.1%}",
                    "✓" if r.is_watermarked else "✗",
                    r.generated_text[:200],
                )

            wandb.log({f"{prefix}steg/samples": table}, step=step)

    def _save_to_jsonl(
        self,
        results: EvaluationResults,
        step: int,
        prefix: str = "",
    ):
        """Save evaluation results to JSONL log file."""
        if self.log_file is None:
            return

        entry = {
            "timestamp": results.timestamp,
            "step": step,
            "prefix": prefix,
            "metrics": {
                "red_alignment": results.red_alignment_avg,
                "blue_alignment": results.blue_alignment_avg,
                "avg_alignment": results.overall_alignment_avg,
                "red_watermark_rate": results.red_watermark_rate,
                "blue_watermark_rate": results.blue_watermark_rate,
                "detection_accuracy": results.detection_accuracy_avg,
            },
            "samples": [
                {
                    "mode": r.mode,
                    "prompt": r.prompt,
                    "alignment": r.alignment,
                    "is_watermarked": r.is_watermarked,
                    "text": r.generated_text,
                    "parity": r.parity,
                }
                for r in results.generation_results
            ],
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")


def create_eval_callback(
    tokenizer,
    eval_every_n_steps: int = 100,
    num_samples: int = 5,
    output_dir: Optional[str] = None,
    run_detection: bool = False,
) -> StegEvalCallback:
    """
    Convenience function to create an evaluation callback.

    Args:
        tokenizer: The tokenizer
        eval_every_n_steps: Evaluation frequency
        num_samples: Samples per mode
        output_dir: Base output directory (logs will go in output_dir/eval_logs/)
        run_detection: Whether to run detection evaluation

    Returns:
        Configured StegEvalCallback
    """
    logs_dir = None
    if output_dir:
        logs_dir = str(Path(output_dir) / "eval_logs")

    return StegEvalCallback(
        tokenizer=tokenizer,
        eval_every_n_steps=eval_every_n_steps,
        num_samples=num_samples,
        logs_dir=logs_dir,
        run_detection=run_detection,
    )
