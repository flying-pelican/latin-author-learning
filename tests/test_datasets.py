import json
from copy import deepcopy
from pathlib import Path

import pytest

from latin_author_learning.datasets import (
    SECTION_SEPARATOR,
    PieceOfWork,
    extract_text,
    read_text,
)


@pytest.fixture(scope="session")
def caesar_quotes():
    return ["Veni, vidi, vici.", "Gallia omnis divisa est in partes tres."]


@pytest.fixture()
def sample_file_content(caesar_quotes):
    return {
        "author": "Gaius Julius Ceasar",
        "name": "famous_quotes",
        "year": "~50 BC",
        "sections": {"pars prima": caesar_quotes[0], "pars secunda": caesar_quotes[1]},
    }


@pytest.fixture()
def sample_json_file(tmpdir, sample_file_content):
    file_name = "caesar_quotes.json"
    file_path = tmpdir / file_name
    with file_path.open("w") as fh:
        json.dump(sample_file_content, fh, indent=4)
    return Path(file_path)


def test_extract_text(sample_file_content, caesar_quotes):
    text = extract_text(sample_file_content, meta_data_keys=["author", "name", "year"])
    assert text == SECTION_SEPARATOR.join(caesar_quotes)


def test_extract_text__multiple_nesting_levels(sample_file_content, caesar_quotes):
    premable = "De rebus diversis."
    commentary = "foo"
    modified_data = {
        "preamble": premable,
        "sections": deepcopy(sample_file_content["sections"]),
        "commentary": commentary,
    }
    text = extract_text(modified_data, meta_data_keys=[])
    assert text == SECTION_SEPARATOR.join([premable] + caesar_quotes + [commentary])


def test_extract_text__list(sample_file_content, caesar_quotes):
    data = deepcopy(sample_file_content)
    data["sections"] = caesar_quotes
    text = extract_text(data, meta_data_keys=["author", "name", "year"])
    assert text == SECTION_SEPARATOR.join(caesar_quotes)


def test_extract_text__wrong_type(sample_file_content):
    modified_data = deepcopy(sample_file_content)
    modified_data["sections"]["pars teria"] = 3

    expected_error_message = f"""
                Got unexpected type {type(3)} from parsed json.
                All values are assumed to be strings.
                """
    with pytest.raises(TypeError, match=expected_error_message):
        _ = extract_text(modified_data, meta_data_keys=["author", "name", "year"])


def test_read_text(sample_json_file, caesar_quotes):
    assert sample_json_file.exists()
    text = read_text(sample_json_file, meta_data_keys=["author", "name", "year"])
    assert text == SECTION_SEPARATOR.join(caesar_quotes)


def test_read_text_wrong_file_format():
    file_path = Path("/tmp") / "sample_file.xml"
    expected_error_message = f"""
            Wrong file format of file {file_path}.
            Currently only JSON files are supported.
            """
    with pytest.raises(NotImplementedError, match=expected_error_message):
        _ = read_text(file_path, meta_data_keys=[])


def test_PieceOfWork():
    author = "Gaius Julius Caesar"
    text = "Veni, vidi, vici."
    name = "quote"

    caesar_quote = PieceOfWork(author=author, text=text, name=name)

    assert caesar_quote.author == author
    assert caesar_quote.text == text
    assert caesar_quote.name == name
    assert caesar_quote.uuid.is_safe
    assert caesar_quote.uuid.version == 4
