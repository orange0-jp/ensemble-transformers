import pytest
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ensemble_transformers.models import EnsembleForVisionEncoderDecoderModel


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def test_inference():
    processor = TrOCRProcessor.from_pretrained(
        'microsoft/trocr-base-handwritten')
    models = [
        VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-handwritten'),
        VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-handwritten'),
    ]

    image = Image.open('tests/examples/a01-122-02.jpg').convert('RGB')
    pixel_values = processor(image, return_tensors='pt').pixel_values
    model = EnsembleForVisionEncoderDecoderModel(
        models=models, config=models[0].config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)
    pixel_values = pixel_values.to(device)

    # greedy_search
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    score = jaccard_similarity(
        generated_text,
        'industry, " Mr. Brown commented icily. " Let us have a')
    assert score > 0.5

    # beam search
    generated_ids = model.generate(
        pixel_values, num_beams=3, early_stopping=True, no_repeat_ngram_size=3)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    score = jaccard_similarity(
        generated_text,
        'industry, " Mr. Brown commented icily. " Let us have a')
    assert score > 0.5

    # sample
    generated_ids = model.generate(
        pixel_values, do_sample=True, top_p=0.95, top_k=0)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    score = jaccard_similarity(
        generated_text,
        'industry, " Mr. Brown commented icily. " Let us have a')
    assert score > 0.5

    # contrastive search
    with pytest.raises(NotImplementedError):
        generated_ids = model.generate(
            pixel_values, penalty_alpha=0.6, top_k=4, use_cache=True)

    # constrained_beam_search
    force_words_ids = [
        processor.tokenizer('industry').input_ids,
    ]
    generated_ids = model.generate(
        pixel_values, num_beams=3, force_words_ids=force_words_ids)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    score = jaccard_similarity(
        generated_text,
        'industry, " Mr. Brown commented icily. " Let us have a')
    assert score > 0.1

    # beam sample
    generated_ids = model.generate(
        pixel_values,
        num_beams=3,
        do_sample=True,
        early_stopping=True,
        no_repeat_ngram_size=3)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    score = jaccard_similarity(
        generated_text,
        'industry, " Mr. Brown commented icily. " Let us have a')
    assert score > 0.5

    # group beam search
    generated_ids = model.generate(
        pixel_values, num_beams=4, num_beam_groups=2, diversity_penalty=10.0)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    score = jaccard_similarity(
        generated_text,
        'industry, " Mr. Brown commented icily. " Let us have a')
    assert score > 0.5
