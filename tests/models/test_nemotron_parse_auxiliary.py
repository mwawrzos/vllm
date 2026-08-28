# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.model_executor.models.nemotron_parse import (
    NemotronParseForConditionalGeneration,
)
from vllm.model_executor.models.nemotron_parse_auxiliary import (
    BoxTokenAuxiliaryHead,
    EndpointDriftDetector,
    NemotronParseAuxiliaryRuntime,
    iter_box_token_auxiliary_weights,
    load_auxiliary_spec,
)
from vllm.model_executor.models.nemotron_parse_worker import (
    NemotronParseGPUModelRunner,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _write_auxiliary_files(tmp_path, *, d_model: int = 2, hidden_size: int = 3):
    tensors = {}
    for branch in ("total_tokens", "remaining_progress"):
        tensors[f"box_token_auxiliary.{branch}.0.weight"] = {
            "shape": [hidden_size, d_model]
        }
        tensors[f"box_token_auxiliary.{branch}.0.bias"] = {"shape": [hidden_size]}
        tensors[f"box_token_auxiliary.{branch}.2.weight"] = {"shape": [1, hidden_size]}
        tensors[f"box_token_auxiliary.{branch}.2.bias"] = {"shape": [1]}

    tensor_file = "auxiliary_prediction_heads.safetensors.extra"
    (tmp_path / "auxiliary_prediction_heads.json").write_text(
        json.dumps(
            {
                "format": "safetensors",
                "tensor_file": tensor_file,
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"vocab": {"<x_0>": 10}},
                "added_tokens": [
                    {"content": "<y_0>", "id": 11},
                    {"content": "<class_Text>", "id": 12},
                ],
            }
        ),
        encoding="utf-8",
    )
    return tensor_file


def _write_auxiliary_sidecar(tmp_path, *, omit: str | None = None):
    head = BoxTokenAuxiliaryHead(2, 1)
    tensors = {
        f"box_token_auxiliary.{name}": torch.ones_like(parameter)
        for name, parameter in head.named_parameters()
        if name != omit
    }
    tensors["decoder.extra_heads.0.weight"] = torch.zeros(2, 2)
    path = tmp_path / "auxiliary_prediction_heads.safetensors.extra"
    save_file(tensors, path)
    return path


def test_auxiliary_head_applies_calibration_transforms():
    head = BoxTokenAuxiliaryHead(d_model=2, hidden_size=1)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.total_tokens[2].bias.fill_(math.log(6.0))

    total, progress = head(torch.zeros(1, 2))

    torch.testing.assert_close(total, torch.tensor([5.0]))
    torch.testing.assert_close(progress, torch.tensor([0.5]))


def test_manifest_defines_head_and_token_groups(tmp_path):
    tensor_file = _write_auxiliary_files(tmp_path)

    assert load_auxiliary_spec(str(tmp_path), None, expected_d_model=2) == (
        tensor_file,
        3,
        {10},
        {11},
        {12},
    )


def test_manifest_rejects_decoder_width_mismatch(tmp_path):
    _write_auxiliary_files(tmp_path, d_model=4)

    with pytest.raises(ValueError, match="hidden size mismatch"):
        load_auxiliary_spec(str(tmp_path), None, expected_d_model=2)


def _auxiliary_model(tmp_path):
    class _WeightSink(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded = []

        def load_weights(self, weights):
            self.loaded.extend(weights)

    model = NemotronParseForConditionalGeneration.__new__(
        NemotronParseForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.encoder = _WeightSink()
    model.decoder = _WeightSink()
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.lm_head.weight.fill_(1)
    model.box_token_auxiliary = BoxTokenAuxiliaryHead(2, 1)
    model._auxiliary_tensor_file = "auxiliary_prediction_heads.safetensors.extra"
    model._auxiliary_model = str(tmp_path)
    model._auxiliary_revision = None
    model._auxiliary_load_config = SimpleNamespace(
        download_dir=None,
        ignore_patterns=None,
        use_tqdm_on_load=False,
        safetensors_load_strategy=None,
    )
    return model


def test_model_loads_suffix_agnostic_safetensors_sidecar(tmp_path):
    _write_auxiliary_sidecar(tmp_path)
    model = _auxiliary_model(tmp_path)

    model.load_weights([])

    assert model.decoder.loaded == []
    assert torch.equal(model.lm_head.weight, torch.ones_like(model.lm_head.weight))
    assert all(
        torch.equal(parameter, torch.ones_like(parameter))
        for parameter in model.box_token_auxiliary.parameters()
    )


def test_model_validates_auxiliary_sidecar_completeness(tmp_path):
    missing = "remaining_progress.2.bias"
    _write_auxiliary_sidecar(tmp_path, omit=missing)
    model = _auxiliary_model(tmp_path)

    with pytest.raises(ValueError, match=missing):
        model.load_weights([])


def test_hf_sidecar_download_uses_exact_manifest_filename(tmp_path, monkeypatch):
    path = _write_auxiliary_sidecar(tmp_path)
    calls = []

    def fake_download(model, cache_dir, allow_patterns, revision, **kwargs):
        calls.append((model, cache_dir, allow_patterns, revision, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(
        "vllm.model_executor.models.nemotron_parse_auxiliary.download_weights_from_hf",
        fake_download,
    )

    weights = dict(
        iter_box_token_auxiliary_weights(
            "org/model",
            "revision",
            path.name,
            download_dir="/cache",
            ignore_patterns=["ignored/*"],
            use_tqdm_on_load=False,
            safetensors_load_strategy=None,
        )
    )

    assert calls == [
        (
            "org/model",
            "/cache",
            ["auxiliary_prediction_heads.safetensors.extra"],
            "revision",
            {"ignore_patterns": ["ignored/*"]},
        )
    ]
    assert weights
    assert all(name.startswith("box_token_auxiliary.") for name in weights)


def test_model_auxiliary_api_matches_runtime():
    model = NemotronParseForConditionalGeneration.__new__(
        NemotronParseForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.box_token_auxiliary = BoxTokenAuxiliaryHead(2, 1)
    model.box_token_auxiliary_token_groups = ({10}, {11}, {12})
    runtime = NemotronParseAuxiliaryRuntime()
    sampling_params = SamplingParams(
        extra_args={"box_token_auxiliary_early_stop": True}
    )
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    runtime.add_request(
        "request",
        sampling_params,
        [1],
        [1],
        model.box_token_auxiliary_token_groups,
        speculative_config=None,
        batch_sharded_sampling=False,
    )

    runtime.consume(
        model,
        ["request"],
        [[10]],
        torch.zeros(1, 2),
        [0, 1],
    )

    assert runtime.requests["request"].generated_tokens == 1


@pytest.mark.parametrize(("json_value", "transported_value"), [(True, 1), (False, 0)])
def test_chat_request_transports_boolean_vllm_xarg_as_integer(
    json_value, transported_value
):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "prompt"}],
            "vllm_xargs": {"box_token_auxiliary_early_stop": json_value},
        }
    )

    value = request.vllm_xargs["box_token_auxiliary_early_stop"]
    assert type(value) is int
    assert value == transported_value


@pytest.mark.parametrize(
    ("transported_value", "enabled"),
    [(True, True), (False, False), (1, True), (0, False)],
)
def test_auxiliary_runtime_accepts_canonical_boolean_transport(
    transported_value, enabled
):
    runtime = NemotronParseAuxiliaryRuntime()
    params = SamplingParams(
        extra_args={"box_token_auxiliary_early_stop": transported_value}
    )
    params.update_from_generation_config({}, eos_token_id=2)

    runtime.add_request(
        "request",
        params,
        [],
        [],
        ({1}, {2}, {3}),
        speculative_config=None,
        batch_sharded_sampling=False,
    )

    assert ("request" in runtime.requests) is enabled


@pytest.mark.parametrize(
    "invalid_value",
    ["true", "false", "", 2, -1, 0.0, 1.0, 0.5],
)
def test_auxiliary_runtime_rejects_ambiguous_truthy_values(invalid_value):
    runtime = NemotronParseAuxiliaryRuntime()

    with pytest.raises(ValueError, match="must be a boolean or its transported 1/0"):
        runtime.add_request(
            "request",
            SamplingParams(
                extra_args={"box_token_auxiliary_early_stop": invalid_value}
            ),
            [],
            [],
            ({1}, {2}, {3}),
            speculative_config=None,
            batch_sharded_sampling=False,
        )


def test_auxiliary_runtime_rejects_speculative_decoding():
    runtime = NemotronParseAuxiliaryRuntime()
    params = SamplingParams(extra_args={"box_token_auxiliary_early_stop": True})
    params.update_from_generation_config({}, eos_token_id=2)

    with pytest.raises(ValueError, match="does not support speculative decoding"):
        runtime.add_request(
            "request",
            params,
            [],
            [],
            ({1}, {2}, {3}),
            speculative_config=object(),
            batch_sharded_sampling=False,
        )


def test_runner_registers_auxiliary_state_after_base_request_reset(monkeypatch):
    runner = NemotronParseGPUModelRunner.__new__(NemotronParseGPUModelRunner)
    runner.nemotron_parse_auxiliary = NemotronParseAuxiliaryRuntime()
    runner.model = SimpleNamespace(box_token_auxiliary_token_groups=({1}, {2}, {3}))
    runner.speculative_config = None
    runner.batch_sharder = None
    params = SamplingParams(extra_args={"box_token_auxiliary_early_stop": True})
    params.update_from_generation_config({}, eos_token_id=9)
    request = SimpleNamespace(
        req_id="request",
        sampling_params=params,
        prompt_token_ids=[10, 11],
        prefill_token_ids=[10, 11],
    )
    scheduler_output = SimpleNamespace(scheduled_new_reqs=[request])

    monkeypatch.setattr(
        GPUModelRunner,
        "_remove_request",
        lambda _runner, _req_id: False,
    )

    def reset_existing_requests(model_runner, output):
        for new_request in output.scheduled_new_reqs:
            model_runner._remove_request(new_request.req_id)

    monkeypatch.setattr(GPUModelRunner, "add_requests", reset_existing_requests)

    runner.add_requests(scheduler_output)

    assert "request" in runner.nemotron_parse_auxiliary.requests


def test_auxiliary_runtime_forces_only_armed_request_eos():
    runtime = NemotronParseAuxiliaryRuntime()
    params = SamplingParams(extra_args={"box_token_auxiliary_early_stop": True})
    params.update_from_generation_config({}, eos_token_id=2)
    for req_id in ("armed", "other"):
        runtime.add_request(
            req_id,
            params,
            [],
            [],
            ({1}, {2}, {3}),
            speculative_config=None,
            batch_sharded_sampling=False,
        )
    runtime.requests["armed"].force_eos = True
    logits = torch.zeros(2, 4)

    runtime.force_pending_eos(logits, ["other", "armed"], [0, 1, 2])

    assert torch.equal(logits[0], torch.zeros(4))
    assert logits[1, 2] == 0
    assert torch.isneginf(logits[1, [0, 1, 3]]).all()


def test_auxiliary_runtime_tracks_trigger_across_batch_reordering():
    class _Model:
        @staticmethod
        def box_token_auxiliary_predictions(hidden_states):
            count = hidden_states.shape[0]
            return torch.ones(count), torch.zeros(count)

    runtime = NemotronParseAuxiliaryRuntime()
    params = SamplingParams(extra_args={"box_token_auxiliary_early_stop": True})
    params.update_from_generation_config({}, eos_token_id=9)
    for req_id in ("armed", "other"):
        runtime.add_request(
            req_id,
            params,
            [],
            [],
            ({1}, {2}, {3}),
            speculative_config=None,
            batch_sharded_sampling=False,
        )
    runtime.requests["armed"].detector.detector_factory = lambda: (
        EndpointDriftDetector(baseline_tokens=1, threshold=0)
    )

    for req_ids, sampled_ids in (
        (["other", "armed"], [[8], [1]]),
        (["armed", "other"], [[2], [8]]),
        (["other", "armed"], [[8], [4]]),
    ):
        runtime.consume(
            _Model(),
            req_ids,
            sampled_ids,
            torch.zeros(2, 2),
            [0, 1, 2],
        )

    assert runtime.requests["armed"].force_eos
    assert not runtime.requests["other"].force_eos
    logits = torch.zeros(2, 10)
    runtime.force_pending_eos(logits, ["other", "armed"], [0, 1, 2])
    assert torch.equal(logits[0], torch.zeros(10))
    assert logits[1].argmax().item() == 9


def test_auxiliary_runtime_preserves_trigger_then_forces_next_eos():
    class _Model:
        @staticmethod
        def box_token_auxiliary_predictions(hidden_states):
            count = hidden_states.shape[0]
            return torch.ones(count), torch.zeros(count)

    runtime = NemotronParseAuxiliaryRuntime()
    params = SamplingParams(extra_args={"box_token_auxiliary_early_stop": True})
    params.update_from_generation_config({}, eos_token_id=9)
    runtime.add_request(
        "request",
        params,
        [],
        [],
        ({1}, {2}, {3}),
        speculative_config=None,
        batch_sharded_sampling=False,
    )
    state = runtime.requests["request"]
    state.detector.detector_factory = lambda: EndpointDriftDetector(
        baseline_tokens=1,
        threshold=0,
    )
    for token_id in (1, 2, 4):
        runtime.consume(
            _Model(),
            ["request"],
            [[token_id]],
            torch.zeros(1, 2),
            [0, 1],
        )

    assert state.force_eos
    logits = torch.zeros(1, 10)
    runtime.force_pending_eos(logits, ["request"], [0, 1])
    assert logits.argmax(dim=-1).tolist() == [9]
