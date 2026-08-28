# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Nemotron Parse box-token auxiliary-head early stopping."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import nn

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    safetensors_weights_iterator,
)
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

AUXILIARY_MANIFEST = "auxiliary_prediction_heads.json"
EARLY_STOP_XARG = "box_token_auxiliary_early_stop"
STOP_REASON = "box_token_auxiliary_drift"
_BOX_TOKEN_AUXILIARY_PREFIX = "box_token_auxiliary."


class BoxTokenAuxiliaryHead(nn.Module):
    """Predict box token count and remaining progress from decoder state."""

    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.total_tokens = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.remaining_progress = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_log = self.total_tokens(hidden_states).squeeze(-1)
        total = torch.expm1(total_log.float())
        progress = torch.sigmoid(
            self.remaining_progress(hidden_states).squeeze(-1).float()
        )
        return total, progress


def load_auxiliary_spec(
    model: str,
    revision: str | None,
    expected_d_model: int | None = None,
) -> tuple[str, int | None, set[int], set[int], set[int]] | None:
    """Load and validate the sidecar manifest and tokenizer token groups."""
    manifest = get_hf_file_to_dict(AUXILIARY_MANIFEST, model, revision)
    if manifest is None:
        return None
    if manifest.get("format") != "safetensors":
        raise ValueError(f"{AUXILIARY_MANIFEST}: expected safetensors format")
    tensor_file = manifest.get("tensor_file")
    tensors = manifest.get("tensors")
    if not isinstance(tensor_file, str) or not isinstance(tensors, dict):
        raise ValueError(f"{AUXILIARY_MANIFEST}: malformed auxiliary manifest")

    first_weight = tensors.get("box_token_auxiliary.total_tokens.0.weight")
    if not isinstance(first_weight, dict):
        return tensor_file, None, set(), set(), set()
    required_tensors = {
        f"box_token_auxiliary.{branch}.{layer}.{parameter}"
        for branch in ("total_tokens", "remaining_progress")
        for layer in (0, 2)
        for parameter in ("weight", "bias")
    }
    missing_tensors = required_tensors - tensors.keys()
    if missing_tensors:
        raise ValueError(
            f"{AUXILIARY_MANIFEST}: incomplete box-token auxiliary head: "
            f"{sorted(missing_tensors)}"
        )
    shape = first_weight.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{AUXILIARY_MANIFEST}: invalid auxiliary head shape")
    hidden_size = int(shape[0])
    if expected_d_model is not None and int(shape[1]) != expected_d_model:
        raise ValueError(
            "box_token_auxiliary hidden size mismatch: "
            f"sidecar={shape[1]}, model={expected_d_model}"
        )

    tokenizer = get_hf_file_to_dict("tokenizer.json", model, revision)
    if tokenizer is None:
        raise ValueError(
            f"{AUXILIARY_MANIFEST}: tokenizer.json is required for box parsing"
        )
    vocabulary: dict[str, int] = {}
    model_vocab = tokenizer.get("model", {}).get("vocab", {})
    if isinstance(model_vocab, dict):
        vocabulary.update(
            (str(token), int(token_id)) for token, token_id in model_vocab.items()
        )
    for token in tokenizer.get("added_tokens", []):
        if isinstance(token, dict) and "content" in token and "id" in token:
            vocabulary[str(token["content"])] = int(token["id"])

    x_ids = {
        token_id
        for token, token_id in vocabulary.items()
        if token.startswith("<x_") and token.endswith(">")
    }
    y_ids = {
        token_id
        for token, token_id in vocabulary.items()
        if token.startswith("<y_") and token.endswith(">")
    }
    class_ids = {
        token_id
        for token, token_id in vocabulary.items()
        if token.startswith("<class_") and token.endswith(">")
    }
    if not x_ids or not y_ids or not class_ids:
        raise ValueError(
            "tokenizer.json: OCR coordinate or class token groups are missing"
        )
    return tensor_file, hidden_size, x_ids, y_ids, class_ids


def iter_box_token_auxiliary_weights(
    model: str,
    revision: str | None,
    tensor_file: str,
    *,
    download_dir: str | None,
    ignore_patterns: str | list[str] | None,
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Load a manifest-declared safetensors sidecar independent of its suffix."""
    if os.path.isdir(model):
        model_folder = model
    else:
        model_folder = download_weights_from_hf(
            model,
            download_dir,
            [tensor_file],
            revision,
            ignore_patterns=ignore_patterns,
        )

    relative_path = Path(tensor_file)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{AUXILIARY_MANIFEST}: tensor_file escapes model directory")
    sidecar_path = Path(model_folder).resolve() / relative_path
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Auxiliary sidecar not found: {sidecar_path}")

    weights = safetensors_weights_iterator(
        [str(sidecar_path)],
        use_tqdm_on_load,
        safetensors_load_strategy,
    )
    yield from (
        (name, tensor)
        for name, tensor in weights
        if name.startswith(_BOX_TOKEN_AUXILIARY_PREFIX)
    )


@dataclass(frozen=True)
class EndpointObservation:
    box_local_token_index: int
    absolute_generated_token_index: int
    T: float
    P: float
    E: float
    B: float | None
    M: float | None
    drift: float | None
    triggered: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema": "box_token_auxiliary_early_stop/v1",
            "stop_reason": STOP_REASON,
            "box_local_token_index": self.box_local_token_index,
            "absolute_generated_token_index": self.absolute_generated_token_index,
            "T": self.T,
            "P": self.P,
            "E": self.E,
            "B": self.B,
            "M": self.M,
            "drift": self.drift,
        }


class EndpointDriftDetector:
    """Exact calibrated E=t+P*(T+4) endpoint-drift detector."""

    baseline_tokens = 100
    alpha = 1.0 - 2.0 ** (-1.0 / 8.0)
    threshold = 1.5

    def __init__(
        self,
        *,
        baseline_tokens: int = 100,
        threshold: float = 1.5,
    ) -> None:
        if baseline_tokens < 1:
            raise ValueError("baseline_tokens must be positive")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        self.baseline_tokens = baseline_tokens
        self.threshold = threshold
        self.baseline_values: list[float] = []
        self.num_observations = 0
        self.B: float | None = None
        self.M: float | None = None
        self.t = 0
        self.valid = True
        self.triggered = False

    def prime_token_count(self, count: int) -> None:
        """Advance box-local indices for prompt tokens without predictions."""
        if count < 0:
            raise ValueError("count must be non-negative")
        self.t += count

    def observe(
        self,
        total: float,
        progress: float,
        absolute_index: int,
        *,
        allow_trigger: bool,
    ) -> EndpointObservation:
        t = self.t
        self.t += 1
        observation_index = self.num_observations
        self.num_observations += 1
        total = float(total)
        progress = float(progress)
        if not math.isfinite(total) or not math.isfinite(progress):
            self.valid = False
        if not self.valid:
            return EndpointObservation(
                t,
                absolute_index,
                total,
                progress,
                math.nan,
                self.B,
                self.M,
                None,
                False,
            )

        endpoint = t + progress * (total + 4.0)
        if observation_index < self.baseline_tokens:
            self.baseline_values.append(endpoint)
            if observation_index == self.baseline_tokens - 1:
                self.B = float(median(self.baseline_values))
                self.M = self.B
            return EndpointObservation(
                t,
                absolute_index,
                total,
                progress,
                endpoint,
                self.B,
                self.M,
                None,
                False,
            )

        assert self.B is not None and self.M is not None
        self.M = self.alpha * endpoint + (1.0 - self.alpha) * self.M
        drift = abs(self.M - self.B) / max(abs(self.B), 1.0)
        triggered = allow_trigger and not self.triggered and drift >= self.threshold
        self.triggered |= triggered
        return EndpointObservation(
            t,
            absolute_index,
            total,
            progress,
            endpoint,
            self.B,
            self.M,
            drift,
            triggered,
        )


class BoxTokenEarlyStopState:
    """Request-scoped alternating coordinate-pair parser and detector."""

    def __init__(
        self,
        x_ids: set[int],
        y_ids: set[int],
        class_ids: set[int],
        detector_factory: Callable[[], EndpointDriftDetector] = (EndpointDriftDetector),
    ) -> None:
        self.x_ids = x_ids
        self.y_ids = y_ids
        self.class_ids = class_ids
        self.detector_factory = detector_factory
        self.detector: EndpointDriftDetector | None = None
        self.inside_box = False
        self.pending_opening_x: tuple[int, float, float] | None = None
        self.pending_prompt_opening_x = False
        self.previous_was_x = False
        self.awaiting_class = False

    def _finish_box(self) -> None:
        self.detector = None
        self.inside_box = False
        self.previous_was_x = False
        self.awaiting_class = False

    def prime_from_prompt_token_ids(self, token_ids: list[int]) -> None:
        """Restore the terminal structured-box state from a request prompt."""
        self.detector = None
        self.inside_box = False
        self.pending_opening_x = None
        self.pending_prompt_opening_x = False
        self.previous_was_x = False
        self.awaiting_class = False

        pending_x = False
        inside_box = False
        previous_was_x = False
        awaiting_class = False
        box_token_count = 0

        for token_id in token_ids:
            if awaiting_class:
                if token_id in self.class_ids:
                    inside_box = False
                    previous_was_x = False
                    awaiting_class = False
                    box_token_count = 0
                    continue
                inside_box = False
                previous_was_x = False
                awaiting_class = False
                box_token_count = 0

            if not inside_box:
                if pending_x:
                    pending_x = False
                    if token_id in self.y_ids:
                        inside_box = True
                        box_token_count = 2
                        continue
                pending_x = token_id in self.x_ids
                continue

            box_token_count += 1
            is_closing_y = previous_was_x and token_id in self.y_ids
            if is_closing_y:
                awaiting_class = True
                previous_was_x = False
            else:
                previous_was_x = token_id in self.x_ids

        if inside_box:
            self.detector = self.detector_factory()
            self.detector.prime_token_count(box_token_count)
            self.inside_box = True
            self.previous_was_x = previous_was_x
            self.awaiting_class = awaiting_class
        elif pending_x:
            self.pending_prompt_opening_x = True

    def _observe(
        self,
        absolute_index: int,
        total: float,
        progress: float,
        *,
        structural_token: bool = False,
    ) -> EndpointObservation | None:
        if self.detector is None:
            return None
        return self.detector.observe(
            total,
            progress,
            absolute_index,
            allow_trigger=not structural_token,
        )

    def consume(
        self,
        token_id: int,
        absolute_index: int,
        total: float,
        progress: float,
    ) -> EndpointObservation | None:
        if self.awaiting_class:
            if token_id in self.class_ids:
                observation = self._observe(
                    absolute_index,
                    total,
                    progress,
                    structural_token=True,
                )
                self._finish_box()
                return observation
            self._finish_box()

        if not self.inside_box:
            if self.pending_prompt_opening_x:
                self.pending_prompt_opening_x = False
                if token_id in self.y_ids:
                    self.detector = self.detector_factory()
                    self.detector.prime_token_count(1)
                    self.inside_box = True
                    return self._observe(
                        absolute_index,
                        total,
                        progress,
                        structural_token=True,
                    )
            if self.pending_opening_x is not None:
                x_index, x_total, x_progress = self.pending_opening_x
                self.pending_opening_x = None
                if token_id in self.y_ids:
                    self.detector = self.detector_factory()
                    self.inside_box = True
                    self._observe(
                        x_index,
                        x_total,
                        x_progress,
                        structural_token=True,
                    )
                    return self._observe(
                        absolute_index,
                        total,
                        progress,
                        structural_token=True,
                    )
            if token_id in self.x_ids:
                self.pending_opening_x = (
                    absolute_index,
                    total,
                    progress,
                )
            return None

        is_closing_y = self.previous_was_x and token_id in self.y_ids
        observation = self._observe(
            absolute_index,
            total,
            progress,
            structural_token=(token_id in self.x_ids or is_closing_y),
        )
        if is_closing_y:
            self.awaiting_class = True
            self.previous_was_x = False
        else:
            self.previous_was_x = token_id in self.x_ids
        return observation


def retain_first_trigger_metadata(
    metadata: dict[str, dict[str, Any]],
    req_id: str,
    observation: EndpointObservation | None,
) -> None:
    """Retain the first causal trigger in a multi-token accepted batch."""
    if observation is not None and observation.triggered:
        metadata.setdefault(req_id, observation.to_metadata())


@dataclass
class _RequestAuxiliaryState:
    detector: BoxTokenEarlyStopState
    generated_tokens: int
    eos_token_id: int
    force_eos: bool = False


class NemotronParseAuxiliaryRuntime:
    """Request-scoped auxiliary stopping for the Nemotron Parse runner."""

    def __init__(self) -> None:
        self.requests: dict[str, _RequestAuxiliaryState] = {}

    def add_request(
        self,
        req_id: str,
        sampling_params: SamplingParams | None,
        prompt_token_ids: list[int] | None,
        prefill_token_ids: list[int] | None,
        token_groups: tuple[set[int], set[int], set[int]] | None,
        *,
        speculative_config: object | None,
        batch_sharded_sampling: bool,
    ) -> None:
        extra_args = sampling_params.extra_args if sampling_params is not None else None
        enabled = extra_args.get(EARLY_STOP_XARG) if extra_args is not None else None
        # ChatCompletionRequest.vllm_xargs currently declares values as
        # ``str | int | float``. Pydantic therefore transports JSON true/false
        # to SamplingParams as the exact integers 1/0. Accept only those two
        # canonical transported values in addition to native booleans; broad
        # truthiness would accidentally enable the feature for strings or
        # arbitrary numbers.
        if type(enabled) is int and enabled in (0, 1):
            enabled = bool(enabled)
        elif enabled is not None and not isinstance(enabled, bool):
            raise ValueError(
                f"{EARLY_STOP_XARG} must be a boolean or its transported 1/0 value"
            )
        if enabled is not True:
            self.remove_request(req_id)
            return
        if speculative_config is not None:
            raise ValueError(f"{EARLY_STOP_XARG} does not support speculative decoding")
        if batch_sharded_sampling:
            raise ValueError(
                f"{EARLY_STOP_XARG} does not support batch-sharded sampling"
            )
        if sampling_params is None or sampling_params.eos_token_id is None:
            raise ValueError(f"{EARLY_STOP_XARG} requires an enabled EOS token")
        if sampling_params.min_tokens:
            raise ValueError(f"{EARLY_STOP_XARG} does not support min_tokens")
        if token_groups is None:
            raise RuntimeError(
                f"{EARLY_STOP_XARG} requested, but the model has no "
                "box-token auxiliary head"
            )

        prompt = prompt_token_ids or []
        prefill = prefill_token_ids or prompt
        detector = BoxTokenEarlyStopState(*token_groups)
        detector.prime_from_prompt_token_ids(prompt)
        self.requests[req_id] = _RequestAuxiliaryState(
            detector=detector,
            generated_tokens=max(len(prefill) - len(prompt), 0),
            eos_token_id=sampling_params.eos_token_id,
        )

    def remove_request(self, req_id: str) -> None:
        self.requests.pop(req_id, None)

    def has_requests(self, req_ids: list[str]) -> bool:
        return any(req_id in self.requests for req_id in req_ids)

    def force_pending_eos(
        self,
        logits: torch.Tensor,
        req_ids: list[str],
        cu_num_logits: Sequence[int],
    ) -> None:
        for req_idx, req_id in enumerate(req_ids):
            state = self.requests.get(req_id)
            if state is None or not state.force_eos:
                continue
            start = int(cu_num_logits[req_idx])
            end = int(cu_num_logits[req_idx + 1])
            if end - start != 1:
                raise RuntimeError(
                    f"{EARLY_STOP_XARG} expected one sampled token per request"
                )
            logits[start:end].fill_(-torch.inf)
            logits[start, state.eos_token_id] = 0

    def consume(
        self,
        model: nn.Module,
        req_ids: list[str],
        sampled_token_ids: list[list[int]],
        sample_hidden_states: torch.Tensor,
        cu_num_logits: Sequence[int],
    ) -> None:
        active_rows: list[int] = []
        active_requests: list[tuple[int, _RequestAuxiliaryState]] = []
        for req_idx, req_id in enumerate(req_ids):
            state = self.requests.get(req_id)
            if state is None or state.force_eos or not sampled_token_ids[req_idx]:
                continue
            start = int(cu_num_logits[req_idx])
            end = int(cu_num_logits[req_idx + 1])
            if end - start != 1 or len(sampled_token_ids[req_idx]) != 1:
                raise RuntimeError(
                    f"{EARLY_STOP_XARG} expected one sampled token per request"
                )
            active_rows.append(start)
            active_requests.append((sampled_token_ids[req_idx][0], state))
        if not active_rows:
            return

        prediction_fn = getattr(model, "box_token_auxiliary_predictions", None)
        if prediction_fn is None:
            raise RuntimeError(
                f"{EARLY_STOP_XARG} requested, but the model has no "
                "box-token auxiliary head"
            )
        row_indices = torch.tensor(
            active_rows, dtype=torch.long, device=sample_hidden_states.device
        )
        total, progress = prediction_fn(sample_hidden_states[row_indices])
        total_values = total.float().cpu().tolist()
        progress_values = progress.float().cpu().tolist()
        for (token_id, state), total_value, progress_value in zip(
            active_requests, total_values, progress_values, strict=True
        ):
            observation = state.detector.consume(
                token_id,
                state.generated_tokens,
                total_value,
                progress_value,
            )
            state.generated_tokens += 1
            if observation is not None and observation.triggered:
                state.force_eos = True
