from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel

from ..generation.utils import EnsembleGenerationMixin


class EnsembleForConditionalGeneration(EnsembleGenerationMixin,
                                       PreTrainedModel):
    r"""
    [`EnsembleForConditionalGeneration`] is a generic ensemble model class for
    ConditionalGeneration.
    """

    def __init__(self, models: List[PreTrainedModel],
                 config: PretrainedConfig):
        super(EnsembleForConditionalGeneration, self).__init__(config)
        self.models = nn.ModuleList(models)

    def can_generate(self) -> bool:
        """Returns whether this model can generate sequences with
        `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with
                    `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten,
        # which is a requirement for generation
        if 'GenerationMixin' in str(self.prepare_inputs_for_generation):
            return False
        return True

    def forward(
        self,
        input_ids: Optional[List[torch.LongTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[Tuple[Tuple[
            torch.FloatTensor]]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        r"""
        Args:
            input_ids
                (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding
                will be ignored by default should you provide it.
            attention_mask
                (`torch.Tensor` of shape `(batch_size, sequence_length)`,
                *optional*):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
                Tuple consists of (`last_hidden_state`,
                *optional*: `hidden_states`, *optional*: `attentions`)
                `last_hidden_state` of shape `(batch_size, sequence_length,
                hidden_size)`, *optional*) is a sequence of
                hidden-states at the output of the last layer of the encoder.
                Used in the cross-attention of the decoder.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*,
                returned when `use_cache=True` is passed or when
                `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length
                `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length,
                embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length,
                embed_size_per_head)`.
                Contains pre-computed hidden-states (key and values in the
                self-attention blocks and in the cross-attention
                blocks) that can be used (see `past_key_values` input) to
                speed up sequential decoding.
                If `past_key_values` are used, the user can optionally input
                only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of
                shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
                inputs_embeds (`torch.FloatTensor` of shape
                `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you
                can choose to directly pass an embedded representation.
                This is useful if you want more control over how to
                convert `input_ids` indices into associated vectors than
                the model's internal embedding lookup matrix.
        """
        outputs = []
        if past_key_values is None:
            past_key_values = [
                past_key_values for _ in range(len(self.models))
            ]
        for m, encoder_output, past_key_value in zip(self.models,
                                                     encoder_outputs,
                                                     past_key_values):
            outputs.append(
                m(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_outputs=encoder_output,
                    past_key_values=past_key_value,
                    **kwargs,
                ))
        return Seq2SeqLMOutput(
            loss=None,
            logits=torch.mean(torch.stack([o.logits for o in outputs]), dim=0),
            past_key_values=[o.past_key_values for o in outputs],
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=[
                o.encoder_last_hidden_state for o in outputs
            ],
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def get_encoder(self):
        return [m.get_encoder() for m in self.models]

    def prepare_inputs_for_generation(
        self,
        *args,
        **kwargs,
    ):
        return self.models[0].prepare_inputs_for_generation(*args, **kwargs)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past
