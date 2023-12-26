import random
import string
from pathlib import Path

import lorem
import pytest
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer

from latin_author_learning.tokenize import (
    DELIMITERS,
    _tokenize_words,
    convert_to_tokens,
    get_subtoken_strings,
)


@pytest.fixture(scope="session")
def path_test_subword_encoder():
    file_dir = Path(__file__).parent
    return file_dir / "data" / "test.subword.encoder"


def random_text(length=500):
    delimiters = "".join(DELIMITERS)
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


def lorem_text(sections=2):
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


@pytest.mark.parametrize("delimiter", ["", ".", "?", "!"])
def test_tokenize_words__delimiters(delimiter):
    word_tokenizer = WordTokenizer()
    sentence = "Veni, vidi, vici" + delimiter
    tokenized = _tokenize_words(sentence, word_tokenizer)
    assert tokenized == _tokenize_words("Veni vidi vici", word_tokenizer)


@pytest.mark.parametrize("whitespace", ["   ", "\n", "\t"])
def test_tokenize_words__whitespace(whitespace):
    word_tokenizer = WordTokenizer()
    sentence = "Veni, vidi, vici.".replace(" ", whitespace)
    tokenized_sentence = _tokenize_words(sentence, word_tokenizer)
    assert tokenized_sentence.split() == tokenized_sentence.split(" ")


@pytest.mark.parametrize("delimiter", [",", ";", ":", ".", "!", "?"])
def test_convert_to_tokens__delimiters(delimiter, tokenized_sample_text):
    assert tokenized_sample_text.count(f" {delimiter} ") == 0


def test_convert_to_tokens__lower_case(tokenized_sample_text):
    assert tokenized_sample_text == tokenized_sample_text.lower()


def test_convert_to_tokens__whitespace(tokenized_sample_text):
    assert tokenized_sample_text.split() == tokenized_sample_text.split(" ")
