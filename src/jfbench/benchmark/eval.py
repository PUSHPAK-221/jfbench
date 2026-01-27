import argparse
import asyncio
from dataclasses import dataclass
import gc
import inspect
import json
from pathlib import Path
import random
from typing import Any
from typing import Awaitable
from typing import cast
from typing import Iterable
from typing import Literal
from typing import Sequence
from typing import TYPE_CHECKING
from typing import TypeVar


if TYPE_CHECKING:
    import torch
else:
    from jfbench.imports import LazyImport

    torch = LazyImport("torch")

import pandas as pd
from tqdm import tqdm

from jfbench.benchmark.build import BenchmarkData
from jfbench.benchmark.build import ConstraintSetName
from jfbench.benchmark.build import get_ifbench_benchmark_data
from jfbench.benchmark.build import get_ifbench_benchmark_data_with_multiple_constraints
from jfbench.llm import extract_reasoning_content
from jfbench.llm import LLMClient


MAX_CONCURRENT_EVALUATIONS = 200
DEFAULT_OUTPUT_DIR = Path("data/benchmark_results")
T = TypeVar("T")


@dataclass
class ModelConfig:
    provider: Literal["openrouter", "local", "vllm"]
    model: str
    model_short: str
    extra_body: dict[str, Any] | None = None


def _parse_model_specs_json(value: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("model_specs_json must be valid JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError("model_specs_json must be a JSON array of objects.")
    specs: list[dict[str, Any]] = []
    for index, spec in enumerate(parsed):
        if not isinstance(spec, dict):
            raise ValueError(f"Each model spec must be an object. Invalid entry at index {index}.")
        specs.append(spec)
    if not specs:
        raise ValueError("model_specs_json must not be empty.")
    return specs


def _instantiate_model_configs(specs: Sequence[dict[str, Any]]) -> list[ModelConfig]:
    return [ModelConfig(**spec) for spec in specs]


def _select_judge_config(judge_model_spec_json: str | None) -> ModelConfig | None:
    if judge_model_spec_json is None:
        return None
    try:
        parsed = json.loads(judge_model_spec_json)
    except json.JSONDecodeError as exc:
        raise ValueError("judge_model_spec_json must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("judge_model_spec_json must be a JSON object.")
    return ModelConfig(**parsed)


def _sanitize_filename_component(value: str) -> str:
    text = value.strip()
    if not text:
        return "unknown"
    sanitized = text.replace("/", "-").replace("\\", "-")
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def _result_file_path(
    output_dir: Path,
    benchmark: str,
    n_constraints: int,
    model_short: str,
    constraint_set: ConstraintSetName,
) -> Path:
    safe_benchmark = _sanitize_filename_component(benchmark)
    safe_model_short = _sanitize_filename_component(model_short)
    safe_constraint_set = _sanitize_filename_component(constraint_set)
    return (
        output_dir
        / f"{safe_benchmark}-{safe_constraint_set}-{n_constraints}-{safe_model_short}.jsonl"
    )


def _build_dataset(
    benchmark: str,
    n_constraints: int,
    n_benchmark_data: int | None,
    seed: int,
    constraint_set: ConstraintSetName,
    judge_config: ModelConfig | None = None,
    ifbench_dataset_path: str | None = None,
) -> list[BenchmarkData]:
    if judge_config is None:
        judge_client = LLMClient()
    else:
        judge_client = LLMClient(
            provider=judge_config.provider,
            model=judge_config.model,
            extra_body=judge_config.extra_body,
        )
    if benchmark != "ifbench":
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if n_constraints == 1:
        data = list(
            get_ifbench_benchmark_data(
                judge_client,
                seed=seed,
                constraint_set=constraint_set,
                dataset_path=ifbench_dataset_path,
            )
        )
        if n_benchmark_data is not None:
            data = data[:n_benchmark_data]
    else:
        if n_benchmark_data is None:
            raise ValueError("When n_constraints > 1, n_benchmark_data must be specified.")
        data = list(
            get_ifbench_benchmark_data_with_multiple_constraints(
                judge_client,
                n_constraints=n_constraints,
                n_benchmark_data=n_benchmark_data,
                seed=seed,
                constraint_set=constraint_set,
                dataset_path=ifbench_dataset_path,
            )
        )
    return data


def _load_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_json(results_path, lines=True)
    except ValueError:
        return pd.DataFrame()


def _parse_comma_separated(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_benchmark_list(benchmark: str | Sequence[str]) -> list[str]:
    if isinstance(benchmark, str):
        candidates = _parse_comma_separated(benchmark)
    else:
        candidates = [item.strip() for item in benchmark if str(item).strip()]
    if not candidates:
        raise ValueError("No benchmarks specified.")
    allowed = {"ifbench"}
    for candidate in candidates:
        if candidate not in allowed:
            raise ValueError(
                f"Unsupported benchmark: {candidate}. Allowed values are {sorted(allowed)}."
            )
    return candidates


def _normalize_n_constraints_list(
    n_constraints: int | str | Sequence[int],
) -> list[int]:
    if isinstance(n_constraints, int):
        values = [n_constraints]
    elif isinstance(n_constraints, str):
        str_items = _parse_comma_separated(n_constraints)
        if not str_items:
            raise ValueError("No n-constraints values specified.")
        values = []
        for item in str_items:
            try:
                values.append(int(item))
            except ValueError as exc:
                raise ValueError(f"Invalid n-constraints value: {item}") from exc
    else:
        values = [int(item) for item in n_constraints]
    if not values:
        raise ValueError("No n-constraints values specified.")
    return values


def _dataset_filter_values(
    dataset_list: Sequence[BenchmarkData],
) -> tuple[set[str], set[int]]:
    prompt_sources = {
        str(getattr(item.meta_data, "prompt_source", "")).strip()
        for item in dataset_list
        if getattr(item.meta_data, "prompt_source", None)
    }
    n_constraints_values = {
        int(getattr(item.meta_data, "n_constraints", 0))
        for item in dataset_list
        if getattr(item.meta_data, "n_constraints", None) is not None
    }
    return prompt_sources, n_constraints_values


def _filter_results_by_dataset(
    frame: pd.DataFrame,
    prompt_sources: set[str],
    n_constraints_values: set[int],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.loc[
        frame["prompt_source"].isin(prompt_sources)
        & frame["n_constraints"].isin(n_constraints_values)
    ]


def _shuffle_and_slice_dataset(
    dataset: Iterable[T],
    limit: int | None,
    seed: int,
) -> list[T]:
    dataset_list = list(dataset)
    if limit is None:
        return dataset_list
    shuffled = list(dataset_list)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def _collect_pending_generation(
    config: ModelConfig,
    dataset_list: list[BenchmarkData],
    override: bool,
    results_path: Path,
) -> list[tuple[int, BenchmarkData]]:
    prompt_sources, n_constraints_values = _dataset_filter_values(dataset_list)
    existing_results = _load_results(results_path)
    completed_indices: set[int] = set()
    if not existing_results.empty:
        mask = existing_results["model_short"] == config.model_short
        model_results = existing_results.loc[mask]
        model_results = _filter_results_by_dataset(
            model_results,
            prompt_sources,
            n_constraints_values,
        )
        completed_indices = set(model_results["prompt_index"].astype(int))
    if override:
        completed_indices = set()
    pending_items = [
        (index, benchmark_data)
        for index, benchmark_data in enumerate(dataset_list)
        if index not in completed_indices
    ]
    skip_count = len(dataset_list) - len(pending_items)
    if skip_count:
        print(
            f"Skipping {skip_count} existing entries for model {config.model} ({config.model_short})."
        )
    if not pending_items:
        print(f"No new prompts to generate for model: {config.model} ({config.model_short})")
    return pending_items


def _collect_pending_evaluations(
    config: ModelConfig,
    dataset_list: list[BenchmarkData],
    override: bool,
    results_path: Path,
) -> list[tuple[int, BenchmarkData, str, Any | None]]:
    if not dataset_list:
        return []
    prompt_sources, n_constraints_values = _dataset_filter_values(dataset_list)
    existing_results = _load_results(results_path)
    if existing_results.empty:
        return []
    mask = existing_results["model_short"] == config.model_short
    model_results = existing_results.loc[mask]
    model_results = _filter_results_by_dataset(
        model_results,
        prompt_sources,
        n_constraints_values,
    )
    if model_results.empty:
        return []
    if override:
        pending_eval = model_results
    else:
        pending_eval = model_results.loc[model_results["results"].isna()]
    evaluation_items: list[tuple[int, BenchmarkData, str, Any | None]] = []
    for row in pending_eval.itertuples():
        if pd.isna(row.response):
            continue
        index = int(row.prompt_index)
        if 0 <= index < len(dataset_list):
            response_details = getattr(row, "response_details", None)
            try:
                if pd.isna(response_details):
                    response_details = None
            except Exception:
                pass
            evaluation_items.append((index, dataset_list[index], row.response, response_details))
    return evaluation_items


async def _evaluate_entries(
    config: ModelConfig,
    evaluation_items: list[tuple[int, BenchmarkData, str, Any | None]],
) -> list[dict[str, Any]]:
    async def _evaluate_entry(
        index: int,
        benchmark_data: BenchmarkData,
        response: str,
        response_details: Any | None,
    ) -> dict[str, Any]:
        evaluation = benchmark_data.evaluate(response)
        if inspect.isawaitable(evaluation):
            awaitable = cast("Awaitable[dict[str, bool]]", evaluation)
            evaluation_dict = await awaitable
        else:
            evaluation_dict = cast("dict[str, bool]", evaluation)
        reasoning_content = extract_reasoning_content(config.provider, response_details)
        return {
            "model": config.model,
            "model_short": config.model_short,
            "prompt_index": index,
            "response": response,
            "response_details": response_details,
            "reasoning_content": reasoning_content,
            "results": evaluation_dict,
            **benchmark_data.meta_data.__dict__,
        }

    tasks = [
        _evaluate_entry(index, benchmark_data, response, response_details)
        for index, benchmark_data, response, response_details in evaluation_items
    ]
    entries: list[dict[str, Any]] = []
    if tasks:
        progress = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Evaluating {config.model}",
            leave=False,
        )
        try:
            for coro in progress:
                entries.append(await coro)
        finally:
            progress.close()
        entries.sort(key=lambda entry: entry["prompt_index"])
    return entries


async def _generate_responses(
    client: LLMClient,
    config: ModelConfig,
    pending_items: list[tuple[int, BenchmarkData]],
    n_concurrent_generations: int = -1,
) -> list[dict[str, Any]]:
    prompts = [benchmark_data.text() for _, benchmark_data in pending_items]
    if not prompts:
        return []

    batch_size = len(prompts)
    if n_concurrent_generations > 0:
        batch_size = min(n_concurrent_generations, len(prompts))

    responses: list[str] = []
    response_details: list[Any] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        if config.provider == "openrouter" or config.provider == "vllm":
            batch_responses, batch_details = await client.async_ask(batch_prompts, use_tqdm=True)
        elif config.provider == "local":
            batch_responses, batch_details = await asyncio.to_thread(
                client.ask, batch_prompts, use_tqdm=True
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        responses.extend(batch_responses)
        response_details.extend(batch_details)

    if len(responses) != len(pending_items):
        raise RuntimeError("Number of responses does not match number of pending prompts.")
    if len(response_details) != len(pending_items):
        raise RuntimeError("Number of response details does not match number of pending prompts.")

    return [
        {
            "model": config.model,
            "model_short": config.model_short,
            "prompt_index": index,
            "response": response,
            "response_details": detail,
            "reasoning_content": extract_reasoning_content(config.provider, detail),
            "results": None,
            **benchmark_data.meta_data.__dict__,
        }
        for (index, benchmark_data), response, detail in zip(
            pending_items, responses, response_details, strict=True
        )
    ]


def _upsert_entries(results_path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_results(results_path)
    new_df = pd.DataFrame(entries)
    if existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
    required_columns = {"model_short", "prompt_index", "prompt_source", "n_constraints"}
    if required_columns <= set(merged.columns):
        merged["prompt_index"] = merged["prompt_index"].astype(int)
        subset = ["model_short", "prompt_index", "prompt_source", "n_constraints"]
        sort_by = list(subset)
        merged = merged.drop_duplicates(subset=subset, keep="last")
        merged = merged.sort_values(by=sort_by).reset_index(drop=True)
    json_lines = merged.to_json(orient="records", lines=True, index=False)
    if json_lines:
        results_path.write_text(json_lines, encoding="utf-8")


async def evaluate_model(
    dataset: Iterable[BenchmarkData],
    config: ModelConfig,
    with_generate: bool,
    with_eval: bool,
    override: bool,
    results_path: Path,
    client: LLMClient | None,
    n_concurrent_generations: int = -1,
) -> tuple[LLMClient | None, list[dict[str, Any]]]:
    dataset_list = list(dataset)
    new_entries: list[dict[str, Any]] = []

    pending_items: list[tuple[int, BenchmarkData]] = []
    if with_generate:
        pending_items = _collect_pending_generation(config, dataset_list, override, results_path)
        if pending_items:
            if client is None:
                client = LLMClient(
                    provider=config.provider,
                    model=config.model,
                    extra_body=config.extra_body,
                )
            generated_entries = await _generate_responses(
                client,
                config,
                pending_items,
                n_concurrent_generations=n_concurrent_generations,
            )
            _upsert_entries(results_path, generated_entries)
            new_entries.extend(generated_entries)

    if with_eval:
        evaluation_items = _collect_pending_evaluations(
            config,
            dataset_list,
            override,
            results_path,
        )
        if evaluation_items:
            evaluated_entries = await _evaluate_entries(config, evaluation_items)
            _upsert_entries(results_path, evaluated_entries)
            new_entries.extend(evaluated_entries)

    return client, new_entries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ifbench",
        help="Benchmark dataset to use. Only 'ifbench' is supported.",
    )
    parser.add_argument(
        "--ifbench-dataset-path",
        type=str,
        default=None,
        help="Optional path to the IFBench dataset JSONL. Defaults to the bundled dataset when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where benchmark results will be written. Each benchmark/model combination "
            "will be stored as its own JSONL file."
        ),
    )
    parser.add_argument(
        "--with-generate",
        dest="with_generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable running the generation step.",
    )
    parser.add_argument(
        "--with-eval",
        dest="with_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable running the evaluation step.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-run generation and evaluation even if entries already exist.",
    )
    parser.add_argument(
        "--n-constraints",
        type=str,
        default="1",
        help="Number of constraints to apply. Provide comma-separated values to run multiple counts.",
    )
    parser.add_argument(
        "--constraint-set",
        type=str,
        choices=["training", "test"],
        default="test",
        help="Choose whether to build with the training or test constraint set.",
    )
    parser.add_argument(
        "--n-benchmark-data",
        type=int,
        default=None,
        help=(
            "Number of benchmark data entries to use. If not set, use all available entries when "
            "n_constraints is 1. If n_constraints > 1, you must set this value."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model-specs-json",
        type=str,
        required=True,
        help="JSON string describing the models to evaluate.",
    )
    parser.add_argument(
        "--judge-model-spec-json",
        type=str,
        default=None,
        help="JSON string describing the model to use for judge_client.",
    )
    parser.add_argument(
        "--n-concurrent-generations",
        type=int,
        default=-1,
        help=(
            "Number of prompts to send concurrently to ask/async_ask. "
            "Use -1 to send all prompts at once."
        ),
    )
    return parser.parse_args(argv)


async def main(
    benchmark: str | Sequence[str] = "ifbench",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    with_generate: bool = True,
    with_eval: bool = True,
    override: bool = False,
    n_constraints: int | str | Sequence[int] = 1,
    n_benchmark_data: int | None = None,
    seed: int = 42,
    model_specs_json: str | None = None,
    judge_model_spec_json: str | None = None,
    constraint_set: ConstraintSetName = "training",
    ifbench_dataset_path: str | None = None,
    n_concurrent_generations: int = -1,
) -> None:
    if model_specs_json is not None:
        custom_specs = _parse_model_specs_json(model_specs_json)
        configs = _instantiate_model_configs(custom_specs)
    else:
        raise ValueError("model_specs_json is required.")
    judge_config = _select_judge_config(judge_model_spec_json)
    benchmark_list = _normalize_benchmark_list(benchmark)
    n_constraints_list = _normalize_n_constraints_list(n_constraints)
    output_dir_path = Path(output_dir)
    # Build datasets
    datasets: dict[tuple[str, int], list[BenchmarkData]] = {}
    for benchmark_name in benchmark_list:
        for n_constraints_value in n_constraints_list:
            print(
                f"Building benchmark: {benchmark_name} with {n_constraints_value} constraints "
                f"using {constraint_set} constraint set."
            )
            dataset = _build_dataset(
                benchmark=benchmark_name,
                n_constraints=n_constraints_value,
                n_benchmark_data=n_benchmark_data,
                seed=seed,
                constraint_set=constraint_set,
                judge_config=judge_config,
                ifbench_dataset_path=ifbench_dataset_path,
            )
            datasets[(benchmark_name, n_constraints_value)] = dataset
    for config in configs:
        print(f"Selected model: {config.model} ({config.model_short})")
        client: LLMClient | None = None
        for benchmark_name in benchmark_list:
            for n_constraints_value in n_constraints_list:
                print(
                    f"Running benchmark: {benchmark_name} with {n_constraints_value} constraints "
                    f"using {constraint_set} constraint set."
                )
                dataset = datasets[(benchmark_name, n_constraints_value)]
                print(f"Loaded {len(dataset)} benchmark data entries.")
                results_path = _result_file_path(
                    output_dir_path,
                    benchmark_name,
                    n_constraints_value,
                    config.model_short,
                    constraint_set,
                )
                client, _ = await evaluate_model(
                    dataset=dataset,
                    config=config,
                    with_generate=with_generate,
                    with_eval=with_eval,
                    override=override,
                    results_path=results_path,
                    client=client,
                    n_concurrent_generations=n_concurrent_generations,
                )
                print(
                    f"Evaluation finished for {benchmark_name} with {n_constraints_value} constraints."
                )
        if client is not None and config.provider == "local":
            del client.client
            gc.collect()
            _empty_torch_cuda_cache()


def _empty_torch_cuda_cache() -> None:
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    if callable(is_available) and not is_available():
        return
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            benchmark=args.benchmark,
            output_dir=args.output_dir,
            with_generate=args.with_generate,
            with_eval=args.with_eval,
            override=args.override,
            n_constraints=args.n_constraints,
            n_benchmark_data=args.n_benchmark_data,
            seed=args.seed,
            model_specs_json=args.model_specs_json,
            judge_model_spec_json=args.judge_model_spec_json,
            constraint_set=args.constraint_set,
            ifbench_dataset_path=args.ifbench_dataset_path,
            n_concurrent_generations=args.n_concurrent_generations,
        )
    )
