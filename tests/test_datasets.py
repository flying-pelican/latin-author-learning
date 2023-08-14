import json
from copy import deepcopy
from pathlib import Path

import pytest

from latin_author_learning.datasets import (
    SECTION_SEPARATOR,
    Corpus,
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
        "@author": "Gaius Julius Ceasar",
        "@name": "famous_quotes",
        "@year": "~50 BC",
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
    text = extract_text(sample_file_content, meta_keys=["@author", "@name", "@year"])
    assert text == SECTION_SEPARATOR.join(caesar_quotes)


def test_extract_text__meta_key_prefix(sample_file_content, caesar_quotes):
    text = extract_text(sample_file_content, meta_key_prefix="@")
    assert text == SECTION_SEPARATOR.join(caesar_quotes)


def test_extract_text__multiple_nesting_levels(sample_file_content, caesar_quotes):
    premable = "De rebus diversis."
    commentary = "foo"
    modified_data = {
        "preamble": premable,
        "sections": deepcopy(sample_file_content["sections"]),
        "commentary": commentary,
    }
    text = extract_text(modified_data)
    assert text == SECTION_SEPARATOR.join([premable] + caesar_quotes + [commentary])


def test_extract_text__list(sample_file_content, caesar_quotes):
    data = deepcopy(sample_file_content)
    data["sections"] = caesar_quotes
    text = extract_text(data, meta_keys=["@author", "@name", "@year"])
    assert text == SECTION_SEPARATOR.join(caesar_quotes)


def test_extract_text__wrong_type(sample_file_content):
    modified_data = deepcopy(sample_file_content)
    modified_data["sections"]["pars teria"] = 3

    expected_error_message = f"""
                Got unexpected type {type(3)} from parsed json.
                All values are assumed to be strings, or lists, or dicts.
                """
    with pytest.raises(TypeError, match=expected_error_message):
        _ = extract_text(modified_data, meta_key_prefix="@")


def test_read_text(sample_json_file, caesar_quotes):
    assert sample_json_file.exists()
    text = read_text(sample_json_file, meta_key_prefix="@")
    assert text == SECTION_SEPARATOR.join(caesar_quotes)


def test_read_text_wrong_file_format():
    file_path = Path("/tmp") / "sample_file.xml"
    expected_error_message = f"""
            Wrong file format of file {file_path}.
            Currently only JSON files are supported.
            """
    with pytest.raises(NotImplementedError, match=expected_error_message):
        _ = read_text(file_path)


def test_PieceOfWork():
    author = "Gaius Julius Caesar"
    text = "Veni, vidi, vici."
    title = "quote"

    caesar_quote = PieceOfWork(author=author, text=text, title=title)

    assert caesar_quote.author == author
    assert caesar_quote.text == text
    assert caesar_quote.title == title
    assert caesar_quote.hash.name == "md5"


def test_PieceOfWork__self_equality():
    famous_quote = PieceOfWork(
        author="Gaius Julius Caesar",
        title="famous quote",
        text="Veni, vidi, vici.",
    )
    assert famous_quote == famous_quote


def test_PieceOfWork__equality_with_meta_data_change():
    famous_quote = PieceOfWork(
        author="Gaius Julius Caesar",
        title="famous quote",
        text="Veni, vidi, vici.",
    )
    arrogant_quote = PieceOfWork(
        author="Caesar",
        title="arrogant quote",
        text=famous_quote.text,
    )
    assert famous_quote == arrogant_quote


def test_PieceOfWork__not_equal():
    famous_quote = PieceOfWork(
        author="Gaius Julius Caesar",
        title="famous quote",
        text="Veni, vidi, vici.",
    )
    changed_quote = PieceOfWork(
        author=famous_quote.author,
        title=famous_quote.title,
        text=famous_quote.text.replace(".", "!"),
    )
    assert famous_quote != changed_quote


def test_PieceOfWork__no_comparison_to_other_types():
    famous_quote = PieceOfWork(
        author="Gaius Julius Caesar",
        title="famous quote",
        text="Veni, vidi, vici.",
    )
    expected_error_message = "Cannot compare instances of types "
    expected_error_message += f"{type(famous_quote)} and {type(famous_quote.text)}."
    with pytest.raises(TypeError, match=expected_error_message):
        famous_quote == famous_quote.text
    with pytest.raises(TypeError, match=expected_error_message):
        famous_quote.text == famous_quote


def test_Corpus():
    name = "test corpus"
    corpus = Corpus(name)
    assert corpus.name == name
    assert len(corpus.works) == 0

    famous_quote = PieceOfWork(
        author="Gaius Julius Caesar",
        title="famous quote",
        text="Veni, vidi, vici.",
    )

    corpus.add_piece_of_work(famous_quote)
    assert corpus.works == set([famous_quote.title])

    de_bello_gallico = PieceOfWork(
        author="Gaius Julius Caesar",
        title="de bello gallico",
        text="Gallia omnis ...",
    )
    corpus.add_piece_of_work(de_bello_gallico)
    assert corpus.works == set([famous_quote.title, de_bello_gallico.title])


def test_Corpus__no_duplicates():
    name = "test corpus"
    corpus = Corpus(name)
    assert corpus.name == name
    assert len(corpus.works) == 0

    famous_quote = PieceOfWork(
        author="Gaius Julius Caesar",
        title="famous quote",
        text="Veni, vidi, vici.",
    )

    corpus.add_piece_of_work(famous_quote)
    assert corpus.works == set([famous_quote.title])

    arrogant_quote = PieceOfWork(
        author=famous_quote.author, title="arrogant quote", text=famous_quote.text
    )
    with pytest.raises(
        ValueError, match=f"Added text idential to text {famous_quote.title}"
    ):
        corpus.add_piece_of_work(arrogant_quote)
