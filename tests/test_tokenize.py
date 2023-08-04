import random
import string
from pathlib import Path

import lorem
import pytest
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer

from latin_author_learning.tokenize import (
    ENDS,
    STARTS,
    WORD_SEPARATOR,
    convert_to_tokens,
    get_subtoken_strings,
)


@pytest.fixture(scope="session")
def path_test_subword_encoder():
    file_dir = Path(__file__).parent
    return file_dir / "test_data" / "test.subword.encoder"


def random_text(length=1000):
    delimiters = ".!?,;"
    symbols = string.ascii_letters + string.whitespace + delimiters
    text = ""
    for _ in range(length):
        symbol = random.choice(symbols)
        if symbol in delimiters:
            symbol += " "
        text += symbol
    return text


def real_latin_text():
    return "Cogito ergo sum. Errare humanum est. Veni, vidi, vici."


@pytest.fixture(scope="session", params=[random_text, real_latin_text, lorem.text])
def sample_text(request):
    return request.param()


@pytest.fixture()
def tokenized_sample_text(sample_text):
    return convert_to_tokens(sample_text)


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
    sentence_num = len(tokenized_sample_text.split(ENDS + WORD_SEPARATOR + STARTS))
    assert sentence_num == len(SentenceTokenizer().tokenize(sample_text))


def test_convert_to_tokens__delimiters(tokenized_sample_text):
    for symbol in [",", ";", "?", "!"]:
        assert tokenized_sample_text.count(symbol) == tokenized_sample_text.split(
            WORD_SEPARATOR
        ).count(symbol)


def test_convert_to_tokens__whitespaces(tokenized_sample_text):
    disallowed_whitespaces = [w for w in string.whitespace if w != " "]
    for symbol in disallowed_whitespaces:
        assert tokenized_sample_text.count(symbol) == 0
