import pytest
import torch
from transformers import MarianMTModel, MarianTokenizer

from ensemble_transformers.models import EnsembleForConditionalGeneration


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def test_inference():
    model_name = 'Helsinki-NLP/opus-mt-en-roa'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    models = [
        MarianMTModel.from_pretrained(model_name),
        MarianMTModel.from_pretrained(model_name),
    ]

    src_text = '>>fra<< this is a sentence in english that we want to translate to french'  # noqa
    input = tokenizer(src_text, return_tensors='pt', padding=True)
    model = EnsembleForConditionalGeneration(
        models=models, config=models[0].config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)
    input_ids = input['input_ids'].to(device)

    # greedy_search
    generated_ids = model.generate(input_ids)
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    score = jaccard_similarity(
        generated_text,
        "c'est une phrase en anglais que nous voulons traduire en français")
    assert score > 0.5

    # beam search
    generated_ids = model.generate(
        input_ids, num_beams=3, early_stopping=True, no_repeat_ngram_size=3)
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    score = jaccard_similarity(
        generated_text,
        "c'est une phrase en anglais que nous voulons traduire en français")
    assert score > 0.5
    # sample
    generated_ids = model.generate(
        input_ids, do_sample=True, top_p=0.95, top_k=0)
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    score = jaccard_similarity(
        generated_text,
        "c'est une phrase en anglais que nous voulons traduire en français")
    assert score > 0.5

    # contrastive search
    with pytest.raises(NotImplementedError):
        generated_ids = model.generate(
            input_ids, penalty_alpha=0.6, top_k=4, num_beams=1, use_cache=True)

    # constrained_beam_search
    force_words_ids = [
        tokenizer('phrase').input_ids,
    ]
    generated_ids = model.generate(
        input_ids, num_beams=3, force_words_ids=force_words_ids)
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    score = jaccard_similarity(
        generated_text,
        "c'est une phrase en anglais que nous voulons traduire en français")
    assert score > 0.1

    # beam sample
    generated_ids = model.generate(
        input_ids,
        num_beams=3,
        do_sample=True,
        early_stopping=True,
        no_repeat_ngram_size=3)
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    score = jaccard_similarity(
        generated_text,
        "c'est une phrase en anglais que nous voulons traduire en français")
    assert score > 0.5

    # group beam search
    generated_ids = model.generate(
        input_ids, num_beams=4, num_beam_groups=2, diversity_penalty=10.0)
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    score = jaccard_similarity(
        generated_text,
        "c'est une phrase en anglais que nous voulons traduire en français")
    assert score > 0.5
