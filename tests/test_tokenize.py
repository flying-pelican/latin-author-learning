import random
import string
from pathlib import Path

import lorem
import pytest
import torch
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer

from latin_author_learning.tokenize import (
    ENDS,
    STARTS,
    WORD_SEPARATOR,
    SentenceAwareEncoder,
    convert_to_tokens,
    get_subtoken_strings,
)


@pytest.fixture(scope="session")
def path_test_subword_encoder():
    file_dir = Path(__file__).parent
    return file_dir / "test_data" / "test.subword.encoder"


def random_text(length=500):
    delimiters = ".!?,;"
    symbols = string.ascii_letters + string.whitespace + delimiters + string.digits
    text = ""
    for _ in range(length):
        symbol = random.choice(symbols)
        if symbol in delimiters:
            symbol += " "
        text += symbol
    return text


def real_latin_text():
    return "Cogito ergo sum. Errare humanum est. Veni, vidi, vici."


def lorem_text(sections=10):
    text = ""
    for _ in range(sections):
        text += lorem.text()
    return text


@pytest.fixture(scope="session", params=[random_text, real_latin_text, lorem_text])
def sample_text(request):
    return request.param()


@pytest.fixture(scope="session")
def tokenized_sample_text(sample_text):
    return convert_to_tokens(sample_text)


@pytest.fixture(scope="session")
def vocabulary(path_test_subword_encoder):
    return get_subtoken_strings(path_test_subword_encoder)


def test_get_substring_tokens(path_test_subword_encoder):
    subtoken_strings = get_subtoken_strings(path_test_subword_encoder)

    for s in subtoken_strings:
        assert '"' not in s
        assert "'" not in s
    with path_test_subword_encoder.open("r") as f:
        number_of_lines = sum(1 for _ in f)
    assert len(subtoken_strings) == number_of_lines


def test_convert_to_tokens__length(sample_text, tokenized_sample_text):
    assert len(tokenized_sample_text) > len(sample_text)


def test_convert_to_tokens__sentences(sample_text, tokenized_sample_text):
    sentences = tokenized_sample_text.split(ENDS + WORD_SEPARATOR + STARTS)
    assert len(sentences) == len(SentenceTokenizer().tokenize(sample_text))


def test_convert_to_tokens__delimiters(tokenized_sample_text):
    for symbol in [",", ";", "?", "!"]:
        assert tokenized_sample_text.count(symbol) == tokenized_sample_text.split(
            WORD_SEPARATOR
        ).count(symbol)


def test_convert_to_tokens__lower_case(tokenized_sample_text):
    tokens_wo_control_sequences = tokenized_sample_text.replace(STARTS, "").replace(
        ENDS, ""
    )
    assert tokens_wo_control_sequences == tokens_wo_control_sequences.lower()


def test_convert_to_tokens__whitespace(tokenized_sample_text):
    disallowed_whitespaces = [w for w in string.whitespace if w != WORD_SEPARATOR]
    for symbol in disallowed_whitespaces:
        assert tokenized_sample_text.count(symbol) == 0


def test_SentenceAwareEncoder__invertible(tokenized_sample_text, vocabulary):
    encoder = SentenceAwareEncoder(vocabulary)
    encoded = encoder.encode(tokenized_sample_text)
    assert not torch.is_floating_point(encoded)
    assert encoder.decode(encoded) == tokenized_sample_text


def test_SentenceAwareEncoder__control_sequences(vocabulary):
    encoder = SentenceAwareEncoder(vocabulary)
    assert len(encoder.encode(STARTS)) == 1
    assert len(encoder.encode(ENDS)) == 1


def test_SentenceAwareEncoder__subword(tokenized_sample_text, vocabulary):
    encoder = SentenceAwareEncoder(vocabulary)
    encoded = encoder.encode(tokenized_sample_text)
    assert len(encoded) > len(tokenized_sample_text.split(WORD_SEPARATOR))
