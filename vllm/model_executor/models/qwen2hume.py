from torch.profiler import record_function
import torch
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from transformers import Qwen2Config
from vllm.config import CacheConfig, LoRAConfig
from typing import List, Optional
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.sequence import IntermediateTensors


class CustomQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__(config, cache_config, quant_config, lora_config)
        self.speaker_embedding_mode = (
            config.speaker_embedding_mode
        )  # NOTE: "add" or "token"
        self.speaker_token_id = config.speaker_token_id
        self.speaker_embedding_size = config.speaker_embedding_size
        self.speaker_token_include_list = config.speaker_token_include_list
        self.normalize_speaker_embedding = config.normalize_speaker_embedding
        if self.speaker_embedding_mode is not None:
            self.speaker_projection = torch.nn.Linear(
                self.speaker_embedding_size, config.hidden_size
            ).to(self.device)

    def _create_inputs_embeddings_token(self, input_ids=None, 
                                        speaker_embeddings=None):
        input_shape = input_ids.size()
        seq_len = input_shape[-1]
        batch_size = input_shape[0]

        input_ids = input_ids.view(batch_size, seq_len)

        inputs_embeds = []
        for batch_idx in range(batch_size):
            current_input = input_ids[batch_idx].clone()
            speaker_pos = (current_input == self.speaker_token_id).nonzero(
                as_tuple=True
            )[0]
            embeds = self.model.embed_tokens(current_input)
            if speaker_embeddings is not None and len(speaker_pos) > 0:
                if self.normalize_speaker_embedding:
                    current_speaker_embd = self.speaker_projection(
                        F.normalize(speaker_embeddings[batch_idx])
                    )
                else:
                    current_speaker_embd = self.speaker_projection(
                        speaker_embeddings[batch_idx]
                    )
                embeds[speaker_pos] = current_speaker_embd[: len(speaker_pos)].to(
                    embeds.dtype
                )
            inputs_embeds.append(embeds)

        return torch.stack(inputs_embeds, dim=0)

    def _create_inputs_embeddings_add(self, input_ids=None, speaker_embeddings=None):
        input_shape = input_ids.size()
        seq_len = input_shape[-1]
        batch_size = input_shape[0]

        input_ids = input_ids.view(batch_size, seq_len)

        inputs_embeds = []
        for batch_idx in range(batch_size):
            current_input = input_ids[batch_idx].clone()
            speaker_pos = (current_input == self.speaker_token_id).nonzero(
                as_tuple=True
            )[0]
            embeds = self.model.embed_tokens(current_input)
            if speaker_embeddings is not None and len(speaker_pos) > 0:
                if self.normalize_speaker_embedding:
                    current_speaker_embd = self.speaker_projection(
                        F.normalize(speaker_embeddings[batch_idx])
                    )
                else:
                    current_speaker_embd = self.speaker_projection(
                        speaker_embeddings[batch_idx]
                    )
                for i, pos in enumerate(speaker_pos):
                    next_pos = speaker_pos[i + 1] if i + 1 < len(speaker_pos) else None
                    mask = torch.tensor(
                        [
                            token.item() in self.speaker_token_include_list
                            for token in current_input[pos:next_pos]
                        ],
                        dtype=torch.bool,
                    )
                    if mask.any():
                        embeds[pos:next_pos][mask] += current_speaker_embd[i].to(
                            embeds.dtype
                        )
            inputs_embeds.append(embeds)
        return torch.stack(inputs_embeds, dim=0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        speaker_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("CustomQwen2ForCausalLM.forward"):
            if (
                speaker_embeddings is not None
                and self.speaker_embedding_mode is not None
            ):
                if self.speaker_embedding_mode == "add":
                    inputs_embeds = self._create_inputs_embeddings_add(
                        input_ids=input_ids,
                        speaker_embeddings=speaker_embeddings,
                    )
                    input_ids = None
                elif self.speaker_embedding_mode == "token":
                    inputs_embeds = self._create_inputs_embeddings_token(
                        input_ids=input_ids,
                        speaker_embeddings=speaker_embeddings,
                    )
                    input_ids = None
                else:
                    raise ValueError(
                        "Invalid speaker_embedding_mode: "
                        f"{self.speaker_embedding_mode}"
                    )
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        return hidden_states
