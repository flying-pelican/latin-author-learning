import random
import string
from pathlib import Path

import lorem
import pytest
import torch
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer

from latin_author_learning.tokenize import (
    CONTROL_SEQUENCES,
    DELIMITERS,
    SENTENCE_DELIMITER,
    WORD_SEPARATOR,
    SentenceAwareEncoder,
    _tokenize_words,
    convert_to_tokens,
    get_subtoken_strings,
)


@pytest.fixture(scope="session")
def path_test_subword_encoder():
    file_dir = Path(__file__).parent
    return file_dir / "test_data" / "test.subword.encoder"


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
    assert tokenized.endswith(SENTENCE_DELIMITER)


@pytest.mark.parametrize("whitespace", ["   ", "\n", "\t"])
def test_tokenize_words__whitespace(whitespace):
    word_tokenizer = WordTokenizer()
    sentence = "Veni, vidi, vici.".replace(" ", whitespace)
    tokenized_sentence = _tokenize_words(sentence, word_tokenizer)

    disallowed_whitespaces = string.whitespace.replace(WORD_SEPARATOR, "")
    for symbol in disallowed_whitespaces:
        assert tokenized_sentence.count(symbol) == 0


@pytest.mark.parametrize("delimiter", DELIMITERS)
def test_convert_to_tokens__delimiters(delimiter, tokenized_sample_text):
    assert tokenized_sample_text.count(delimiter + SENTENCE_DELIMITER) == 0


@pytest.mark.parametrize("control_sequence", CONTROL_SEQUENCES)
def test_convert_to_tokens__control_sequences(control_sequence, tokenized_sample_text):
    padded_control_sequence = WORD_SEPARATOR + control_sequence + WORD_SEPARATOR
    assert tokenized_sample_text.count(control_sequence) == tokenized_sample_text.count(
        padded_control_sequence
    )


def test_convert_to_tokens__sentences(sample_text, tokenized_sample_text):
    sentence_count = 0
    for delimiter in CONTROL_SEQUENCES:
        sentence_count += tokenized_sample_text.count(delimiter)
    sentences = SentenceTokenizer().tokenize(sample_text)
    assert len(sentences) == sentence_count


def test_convert_to_tokens__lower_case(tokenized_sample_text):
    text_wo_control_sequences = tokenized_sample_text
    for delimiter in CONTROL_SEQUENCES:
        text_wo_control_sequences = text_wo_control_sequences.replace(
            delimiter, WORD_SEPARATOR
        )
    assert text_wo_control_sequences == text_wo_control_sequences.lower()


def test_convert_to_tokens__whitespace(tokenized_sample_text):
    disallowed_whitespaces = string.whitespace.replace(WORD_SEPARATOR, "")
    for symbol in disallowed_whitespaces:
        assert tokenized_sample_text.count(symbol) == 0


def test_SentenceAwareEncoder__invertible(tokenized_sample_text, vocabulary):
    encoder = SentenceAwareEncoder(vocabulary)
    encoded = encoder.encode(tokenized_sample_text)
    assert not torch.is_floating_point(encoded)
    assert encoder.decode(encoded) == tokenized_sample_text


def test_SentenceAwareEncoder__subword(tokenized_sample_text, vocabulary):
    encoder = SentenceAwareEncoder(vocabulary)
    encoded = encoder.encode(tokenized_sample_text)

    num_words = len(tokenized_sample_text.split(WORD_SEPARATOR))
    assert len(encoded) > num_words


@pytest.mark.parametrize("control_sequence", CONTROL_SEQUENCES)
def test_SentenceAwareEncoder__additional_vocabulary(control_sequence, vocabulary):
    encoder = SentenceAwareEncoder(vocabulary)
    encoded_delimiter = encoder.encode(control_sequence)
    assert len(encoded_delimiter) == 1


@pytest.mark.parametrize("control_sequence", CONTROL_SEQUENCES)
def test_SentenceAwareEncoder__control_sequences(control_sequence, vocabulary):
    encoder = SentenceAwareEncoder(vocabulary)

    sample_text = f"veni {control_sequence} vidi"
    encoded = encoder.encode(sample_text)
    assert len(encoded) == len(sample_text.split(" "))
