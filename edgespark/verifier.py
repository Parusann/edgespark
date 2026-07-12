"""The verifier: an unmodified Qwen3 model that owns every output token.

The verifier is used *as-is*, we never retrain it (spec section 4). Its two jobs:

1. Expose selected intermediate-layer hidden states so the drafter can condition
   on them (``forward_with_hidden``).
2. Verify a drafted block against its own distribution under the exact acceptance
   rule (``verify``), so the emitted sequence is identical to the verifier
   decoding alone (spec section 5).

Whatever precision the verifier ships at *is* the exactness reference. If you
quantize the verifier to free VRAM, exactness is defined against that quantized
verifier, not the fp16 original.

torch + transformers are imported lazily so the rest of the package (and the
numpy test suite) works without them installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from edgespark.loop.acceptance import AcceptResult, verify_block


@dataclass
class VerifierConfig:
    model_name: str = "Qwen/Qwen3-4B"
    precision: str = "fp16"  # 'fp16' | 'int8' | 'nf4' | 'gguf-q4'
    target_layer_ids: tuple[int, ...] = (1, 17, 33)
    device: str = "cuda"
    attn_impl: str = "sdpa"  # 'sdpa' is the safe RDNA3 default; 'flash' needs the navi fork


class Verifier:
    """HF Transformers wrapper with hidden-state capture and exact verification."""

    def __init__(self, config: VerifierConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._captured: dict[int, np.ndarray] = {}
        self._handles: list = []

    # -- lifecycle ------------------------------------------------------------

    def load(self) -> Verifier:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_kwargs = self._quant_kwargs()
        dtype = torch.float16 if self.config.precision == "fp16" else None
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa" if self.config.attn_impl == "sdpa" else "flash_attention_2",
            device_map=self.config.device,
            output_hidden_states=True,
            **quant_kwargs,
        )
        self._model.eval()
        self.hidden_size = self._model.config.hidden_size
        self.vocab_size = self._model.config.vocab_size
        return self

    def _quant_kwargs(self) -> dict:
        # Verifier quantization is optional (spec section 9.3). INT8 / NF4 via
        # bitsandbytes-ROCm; GGUF Q4 is handled by the llama.cpp baseline instead.
        if self.config.precision == "int8":
            from transformers import BitsAndBytesConfig

            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        if self.config.precision == "nf4":
            import torch
            from transformers import BitsAndBytesConfig

            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        return {}

    # -- forward passes -------------------------------------------------------

    def forward_with_hidden(self, tokens, past_key_values=None):
        """Run the verifier and return ``(logits, selected_hidden_states, kv)``.

        ``selected_hidden_states`` is a dict keyed by layer id (the drafter's
        fusion sources). Only the layers in ``target_layer_ids`` are kept, so we
        never hold the full hidden-state stack.
        """
        import torch

        with torch.no_grad():
            out = self._model(
                input_ids=tokens,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        hidden = {
            layer_id: out.hidden_states[layer_id]
            for layer_id in self.config.target_layer_ids
        }
        return out.logits, hidden, out.past_key_values

    def block_distribution(self, prefix_tokens, drafted_block, ell: int):
        """Verifier probability block for the first ``ell`` drafts + bonus.

        Returns an ``[ell + 1, vocab]`` numpy array of probabilities: the exact
        input ``verify_block`` consumes. The verifier scores the whole drafted
        block in a single forward pass, that single pass is the entire point of
        speculative decoding.
        """
        import torch

        seq = torch.cat([prefix_tokens, drafted_block[:ell].unsqueeze(0)], dim=1) \
            if drafted_block.dim() == 1 else torch.cat([prefix_tokens, drafted_block[:, :ell]], dim=1)
        with torch.no_grad():
            logits = self._model(input_ids=seq, use_cache=False).logits[0]
        # The distributions that check draft positions 0..ell-1 plus the bonus are
        # the logits at the last ell+1 sequence positions.
        tail = logits[-(ell + 1):]
        probs = torch.softmax(tail.float(), dim=-1)
        return probs.cpu().numpy()

    def verify(
        self,
        prefix_tokens,
        drafted_block,
        ell: int,
        *,
        mode: str = "greedy",
        draft_dist: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> AcceptResult:
        """Exact accept/reject of ``ell`` drafted tokens. ``ell`` only bounds length."""
        target = self.block_distribution(prefix_tokens, drafted_block, ell)
        heads = drafted_block[:ell].detach().cpu().numpy() if hasattr(drafted_block, "detach") \
            else np.asarray(drafted_block)[:ell]
        return verify_block(target, heads, ell, mode=mode, draft_dist=draft_dist, rng=rng)

    # -- tokenizer helpers ----------------------------------------------------

    def encode(self, text: str):

        ids = self._tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.config.device)

    def decode(self, tokens: Sequence[int]) -> str:
        return self._tokenizer.decode(list(tokens), skip_special_tokens=True)
