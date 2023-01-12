from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel

from ..generation.utils import EnsembleGenerationMixin


class EnsembleForVisionEncoderDecoderModel(EnsembleGenerationMixin,
                                           PreTrainedModel):
    main_input_name = 'pixel_values'

    def __init__(self, models: List[PreTrainedModel],
                 config: PretrainedConfig):
        super(EnsembleForVisionEncoderDecoderModel, self).__init__(config)
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
        pixel_values: Optional[List[torch.LongTensor]] = None,
        encoder_outputs: Optional[List[Tuple[Tuple[
            torch.FloatTensor]]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
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
                    pixel_values=pixel_values,
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

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.models[0]._reorder_cache(past, beam_idx)
