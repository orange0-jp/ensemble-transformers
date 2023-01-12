# flake8: noqa:CCR001,C901
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.generation.beam_search import (BeamScorer,
                                                 ConstrainedBeamSearchScorer)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList, validate_stopping_criteria)
from transformers.generation.utils import (BeamSampleDecoderOnlyOutput,
                                           BeamSampleEncoderDecoderOutput,
                                           BeamSampleOutput,
                                           BeamSearchDecoderOnlyOutput,
                                           BeamSearchEncoderDecoderOutput,
                                           BeamSearchOutput, GenerationMixin)
from transformers.models.auto import (MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
                                      MODEL_FOR_CAUSAL_LM_MAPPING,
                                      MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                                      MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
                                      MODEL_FOR_VISION_2_SEQ_MAPPING)
from transformers.pytorch_utils import torch_int_div
from transformers.utils import ModelOutput


class EnsembleGenerationMixin(GenerationMixin):

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: Optional[torch.LongTensor] = None,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        if model_kwargs.get('token_type_ids') is not None:
            model_kwargs['token_type_ids'] = model_kwargs[
                'token_type_ids'].repeat_interleave(
                    expand_size, dim=0)

        if model_kwargs.get('attention_mask') is not None:
            model_kwargs['attention_mask'] = model_kwargs[
                'attention_mask'].repeat_interleave(
                    expand_size, dim=0)

        if is_encoder_decoder:
            encoder_outputs = model_kwargs.get('encoder_outputs')
            if encoder_outputs is None:
                raise ValueError(
                    'If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.'
                )
            for encoder_output in encoder_outputs:
                encoder_output[
                    'last_hidden_state'] = encoder_output.last_hidden_state.repeat_interleave(
                        expand_size, dim=0)
            model_kwargs['encoder_outputs'] = encoder_outputs

        return input_ids, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            model_kwargs,
            model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        encoders = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs['return_dict'] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs['encoder_outputs'] = [
            encoder(**encoder_kwargs) for encoder in encoders
        ]

        return model_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs['past_key_values'] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format)

        # update token_type_ids with last value
        if 'token_type_ids' in model_kwargs:
            token_type_ids = model_kwargs['token_type_ids']
            model_kwargs['token_type_ids'] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if 'attention_mask' in model_kwargs:
                attention_mask = model_kwargs['attention_mask']
                model_kwargs['attention_mask'] = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ],
                                                           dim=-1)
        else:
            # update decoder attention mask
            if 'decoder_attention_mask' in model_kwargs:
                decoder_attention_mask = model_kwargs['decoder_attention_mask']
                model_kwargs['decoder_attention_mask'] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1))
                    ],
                    dim=-1,
                )

        return model_kwargs

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
        )
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(
        )
        if max_length is not None:
            warnings.warn(
                '`max_length` is deprecated in this function, use'
                ' `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.',
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.generation_config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.generation_config.output_hidden_states)
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else
            self.generation_config.return_dict_in_generate)

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.'
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if
            (return_dict_in_generate and output_scores) else None)
        decoder_attentions = () if (return_dict_in_generate
                                    and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate
                                  and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate
                                       and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs['encoder_outputs'].get(
                'attentions') if output_attentions else None
            encoder_hidden_states = (
                model_kwargs['encoder_outputs'].get('hidden_states')
                if output_hidden_states else None)

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams),
                                  dtype=torch.float,
                                  device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams, ))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits,
                dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(
                input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed, )
                if output_attentions:
                    decoder_attentions += ((outputs.decoder_attentions, ) if
                                           self.config.is_encoder_decoder else
                                           (outputs.attentions, ))
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions, )

                if output_hidden_states:
                    decoder_hidden_states += ((outputs.decoder_hidden_states, )
                                              if self.config.is_encoder_decoder
                                              else (outputs.hidden_states, ))

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size,
                                                       num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores,
                2 * num_beams,
                dim=1,
                largest=True,
                sorted=True)

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs['next_beam_scores']
            beam_next_tokens = beam_outputs['next_beam_tokens']
            beam_idx = beam_outputs['next_beam_indices']

            input_ids = torch.cat(
                [input_ids[beam_idx, :],
                 beam_next_tokens.unsqueeze(-1)],
                dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder)
            if (model_kwargs['past_key_values'] is not None) and (
                    not model_kwargs['past_key_values'][0] is None):
                for i in range(len(model_kwargs['past_key_values'])):
                    model_kwargs['past_key_values'][i] = self._reorder_cache(
                        model_kwargs['past_key_values'][i], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    (beam_indices[beam_idx[i]] + (beam_idx[i], )
                     for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs['sequence_scores'] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    beam_indices=sequence_outputs['beam_indices'],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    beam_indices=sequence_outputs['beam_indices'],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs['sequences']

    def constrained_beam_search(
        self,
        input_ids: torch.LongTensor,
        constrained_beam_scorer: ConstrainedBeamSearchScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **constrained beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation, while satisfying a list of positive constraints. For more information, the
                documentation of [`ConstrainedBeamSearchScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
        )
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(
        )
        if max_length is not None:
            warnings.warn(
                '`max_length` is deprecated in this function, use'
                ' `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.',
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.generation_config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.generation_config.output_hidden_states)
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else
            self.generation_config.return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate
                                    and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate
                                  and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate
                                       and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs['encoder_outputs'].get(
                'attentions') if output_attentions else None
            encoder_hidden_states = (
                model_kwargs['encoder_outputs'].get('hidden_states')
                if output_hidden_states else None)

        batch_size = len(constrained_beam_scorer._beam_hyps)
        num_beams = constrained_beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.'
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams),
                                  dtype=torch.float,
                                  device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams, ))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits,
                dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(
                input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores)

            scores_for_all_vocab = next_token_scores.clone()

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores, )
                if output_attentions:
                    decoder_attentions += ((outputs.decoder_attentions, ) if
                                           self.config.is_encoder_decoder else
                                           (outputs.attentions, ))
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions, )

                if output_hidden_states:
                    decoder_hidden_states += ((outputs.decoder_hidden_states, )
                                              if self.config.is_encoder_decoder
                                              else (outputs.hidden_states, ))

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size,
                                                       num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores,
                2 * num_beams,
                dim=1,
                largest=True,
                sorted=True)

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = constrained_beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs['next_beam_scores']
            beam_next_tokens = beam_outputs['next_beam_tokens']
            beam_idx = beam_outputs['next_beam_indices']

            input_ids = torch.cat(
                [input_ids[beam_idx, :],
                 beam_next_tokens.unsqueeze(-1)],
                dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder)
            if (model_kwargs['past_key_values'] is not None) and (
                    not model_kwargs['past_key_values'][0] is None):
                for i in range(len(model_kwargs['past_key_values'])):
                    model_kwargs['past_key_values'][i] = self._reorder_cache(
                        model_kwargs['past_key_values'][i], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if constrained_beam_scorer.is_done or stopping_criteria(
                    input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = constrained_beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs['sequence_scores'] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs['sequences']

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search multinomial
        sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`~generation.BeamSampleDecoderOnlyOutput`], [`~generation.BeamSampleEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
        )
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(
        )
        if max_length is not None:
            warnings.warn(
                '`max_length` is deprecated in this function, use'
                ' `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.',
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.generation_config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.generation_config.output_hidden_states)
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else
            self.generation_config.return_dict_in_generate)

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if
            (return_dict_in_generate and output_scores) else None)
        decoder_attentions = () if (return_dict_in_generate
                                    and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate
                                  and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate
                                       and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs['encoder_outputs'].get(
                'attentions') if output_attentions else None
            encoder_hidden_states = (
                model_kwargs['encoder_outputs'].get('hidden_states')
                if output_hidden_states else None)

        beam_scores = torch.zeros((batch_size, num_beams),
                                  dtype=torch.float,
                                  device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams, ))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits,
                dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(
                input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (logits_warper(input_ids,
                                             next_token_scores_processed), )
                if output_attentions:
                    decoder_attentions += ((outputs.decoder_attentions, ) if
                                           self.config.is_encoder_decoder else
                                           (outputs.attentions, ))
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions, )

                if output_hidden_states:
                    decoder_hidden_states += ((outputs.decoder_hidden_states, )
                                              if self.config.is_encoder_decoder
                                              else (outputs.hidden_states, ))

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size,
                                                       num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1,
                                             next_tokens)

            next_token_scores, _indices = torch.sort(
                next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )
            beam_scores = beam_outputs['next_beam_scores']
            beam_next_tokens = beam_outputs['next_beam_tokens']
            beam_idx = beam_outputs['next_beam_indices']

            input_ids = torch.cat(
                [input_ids[beam_idx, :],
                 beam_next_tokens.unsqueeze(-1)],
                dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder)
            if (model_kwargs['past_key_values'] is not None) and (
                    not model_kwargs['past_key_values'][0] is None):
                for i in range(len(model_kwargs['past_key_values'])):
                    model_kwargs['past_key_values'][i] = self._reorder_cache(
                        model_kwargs['past_key_values'][i], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    (beam_indices[beam_idx[i]] + (beam_idx[i], )
                     for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs['sequence_scores'] = None

            if self.config.is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    beam_indices=sequence_outputs['beam_indices'],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    beam_indices=sequence_outputs['beam_indices'],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs['sequences']

    def group_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **diverse beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`~generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if [`~generation.BeamSearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
            [`~generation.BeamSearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
        )
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(
        )
        if max_length is not None:
            warnings.warn(
                '`max_length` is deprecated in this function, use'
                ' `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.',
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.generation_config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.generation_config.output_hidden_states)
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else
            self.generation_config.return_dict_in_generate)

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        device = input_ids.device

        batch_beam_size, cur_len = input_ids.shape

        if return_dict_in_generate and output_scores:
            beam_indices = [
                tuple(() for _ in range(num_sub_beams * batch_size))
                for _ in range(num_beam_groups)
            ]
        else:
            beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.'
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate
                                    and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate
                                  and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate
                                       and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs['encoder_outputs'].get(
                'attentions') if output_attentions else None
            encoder_hidden_states = (
                model_kwargs['encoder_outputs'].get('hidden_states')
                if output_hidden_states else None)

        # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores = torch.full((batch_size, num_beams),
                                 -1e9,
                                 dtype=torch.float,
                                 device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams, ))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(
                batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(
                batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            if output_scores:
                processed_score = torch.zeros_like(outputs.logits[:, -1, :])

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend([
                        batch_idx * num_beams + idx
                        for idx in range(group_start_idx, group_end_idx)
                    ])
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of current group only
                next_token_logits = outputs.logits[batch_group_indices, -1, :]

                # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
                # cannot be generated both before and after the `nn.functional.log_softmax` operation.
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len)
                next_token_scores = nn.functional.log_softmax(
                    next_token_logits,
                    dim=-1)  # (batch_size * group_size, vocab_size)
                vocab_size = next_token_scores.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids,
                    next_token_scores,
                    current_tokens=current_tokens,
                    beam_group_idx=beam_group_idx)
                next_token_scores = next_token_scores_processed + beam_scores[
                    batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(
                    next_token_scores_processed)

                if output_scores:
                    processed_score[
                        batch_group_indices] = next_token_scores_processed

                # reshape for beam search
                next_token_scores = next_token_scores.view(
                    batch_size, group_size * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores,
                    2 * group_size,
                    dim=1,
                    largest=True,
                    sorted=True)

                next_indices = torch_int_div(next_tokens, vocab_size)
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(
                    beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                )
                beam_scores[batch_group_indices] = beam_outputs[
                    'next_beam_scores']
                beam_next_tokens = beam_outputs['next_beam_tokens']
                beam_idx = beam_outputs['next_beam_indices']

                if return_dict_in_generate and output_scores:
                    beam_indices[beam_group_idx] = tuple(
                        beam_indices[beam_group_idx][beam_idx[i]] +
                        (beam_idx[i], ) for i in range(len(beam_indices[0])))

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([
                    group_input_ids[beam_idx, :],
                    beam_next_tokens.unsqueeze(-1)
                ],
                                            dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch_int_div(beam_idx, group_size) +
                    group_start_idx + (beam_idx % group_size))

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (processed_score, )
                if output_attentions:
                    decoder_attentions += ((outputs.decoder_attentions, ) if
                                           self.config.is_encoder_decoder else
                                           (outputs.attentions, ))
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions, )

                if output_hidden_states:
                    decoder_hidden_states += ((outputs.decoder_hidden_states, )
                                              if self.config.is_encoder_decoder
                                              else (outputs.hidden_states, ))

            input_ids = torch.cat(
                [input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder)
            if (model_kwargs['past_key_values'] is not None) and (
                    not model_kwargs['past_key_values'][0] is None):
                for i in range(len(model_kwargs['past_key_values'])):
                    model_kwargs['past_key_values'][i] = self._reorder_cache(
                        model_kwargs['past_key_values'][i], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        final_beam_indices = sum(beam_indices,
                                 ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs['sequence_scores'] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    beam_indices=sequence_outputs['beam_indices'],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    beam_indices=sequence_outputs['beam_indices'],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs['sequences']

    def contrastive_search(self, *args, **kwargs):
        raise NotImplementedError

    def _validate_model_class(self):
        """Confirms that the model class is compatible with generation.

        If not, raises an exception that points to the right class to use.
        """
        if not self.can_generate():
            generate_compatible_mappings = [
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
                MODEL_FOR_VISION_2_SEQ_MAPPING,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(
                    type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            generate_compatible_classes.add('EnsembleForConditionalGeneration')
            exception_message = (
                f'The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as '
                "it doesn't have a language model head.")
            if generate_compatible_classes:
                exception_message += f' Please use one of the following classes instead: {generate_compatible_classes}'
            raise TypeError(exception_message)
