# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.nemotron_parse_auxiliary import (
    NemotronParseAuxiliaryRuntime,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu_worker import Worker


class NemotronParseGPUModelRunner(GPUModelRunner):
    """V2 runner with request-scoped Nemotron Parse auxiliary stopping."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nemotron_parse_auxiliary = NemotronParseAuxiliaryRuntime()

    def _remove_request(self, req_id: str) -> bool:
        self.nemotron_parse_auxiliary.remove_request(req_id)
        return super()._remove_request(req_id)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        # GPUModelRunner.add_requests() begins by calling self._remove_request()
        # for every incoming request so resumed/streaming state is rebuilt.
        # Register auxiliary state only after that virtual call; otherwise our
        # _remove_request override immediately deletes the state just added.
        super().add_requests(scheduler_output)
        token_groups = getattr(
            getattr(self, "model", None),
            "box_token_auxiliary_token_groups",
            None,
        )
        for request in scheduler_output.scheduled_new_reqs:
            self.nemotron_parse_auxiliary.add_request(
                request.req_id,
                request.sampling_params,
                request.prompt_token_ids,
                request.prefill_token_ids,
                token_groups,
                speculative_config=self.speculative_config,
                batch_sharded_sampling=self.batch_sharder is not None,
            )

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        runtime = self.nemotron_parse_auxiliary
        if not runtime.has_requests(input_batch.req_ids):
            return super().sample(hidden_states, input_batch, grammar_output)

        if self.batch_sharder is not None or self.rejection_sampler is not None:
            raise RuntimeError(
                "Nemotron Parse auxiliary stopping requires non-speculative, "
                "non-sharded sampling"
            )
        # Each logits index selects the decoder state that causally predicts
        # the token sampled below. Pairing that state with the sampled token
        # matches offline replay's one-token shift (prompt_len - 1 onward).
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )
        runtime.force_pending_eos(
            logits,
            input_batch.req_ids,
            input_batch.cu_num_logits_np,
        )
        assert self.sampler is not None
        sampler_output = self.sampler(logits, input_batch)
        num_sampled = sampler_output.num_sampled
        sampled_rows = sampler_output.sampled_token_ids.cpu().tolist()
        sampled_counts = num_sampled.cpu().tolist()
        sampled_token_ids = [
            row[:count] for row, count in zip(sampled_rows, sampled_counts, strict=True)
        ]
        runtime.consume(
            self.model,
            input_batch.req_ids,
            sampled_token_ids,
            sample_hidden_states,
            input_batch.cu_num_logits_np,
        )
        return sampler_output, num_sampled, sampler_output.num_rejected


class NemotronParseWorker(Worker):
    """GPU worker selectable with ``--worker-cls``."""

    def init_device(self) -> None:
        super().init_device()
        if not self.use_v2_model_runner:
            raise ValueError("NemotronParseWorker requires the V2 GPU model runner")
        self.model_runner = NemotronParseGPUModelRunner(
            self.vllm_config,
            self.device,
        )
