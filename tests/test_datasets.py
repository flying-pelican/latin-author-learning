import json
import string
from copy import deepcopy
from pathlib import Path

import pytest

from latin_author_learning.datasets import (
    SECTION_SEPARATOR,
    Corpus,
    PieceOfWork,
    convert_to_path_name,
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
        "@name": "caesar_quotes",
        "@year": "~50 BC",
        "sections": {
            "pars prima": caesar_quotes[0],
            "pars secunda": caesar_quotes[1],
            "commentario": None,
        },
    }


@pytest.fixture()
def sample_json_file(tmpdir, sample_file_content):
    file_name = "caesar_quotes.json"
    file_path = tmpdir / file_name
    with file_path.open("w") as fh:
        json.dump(sample_file_content, fh, indent=4)
    return Path(file_path)


@pytest.fixture(scope="session")
def caesar_quote():
    return PieceOfWork(
        author="Gaius Julius Caesar",
        title="famous quote",
        text="Veni, vidi, vici.",
    )


@pytest.fixture(scope="session")
def cato_quote():
    return PieceOfWork(
        author="Cato senior", title="famous quote", text="Ceterum censeo ..."
    )


@pytest.fixture(scope="session")
def de_bello_gallico():
    return PieceOfWork(
        author="Gaius Julius Caesar",
        title="de bello gallico",
        text="Gallia omnis ...",
    )


@pytest.fixture(scope="session")
def sample_corpus(caesar_quote, de_bello_gallico, cato_quote):
    corpus = Corpus(name="sample corpus")
    corpus.add_piece_of_work(caesar_quote)
    corpus.add_piece_of_work(de_bello_gallico)
    corpus.add_piece_of_work(cato_quote)
    return corpus


@pytest.fixture(scope="session", params=[None, "opensource"])
def sub_folder(request):
    return request.param


@pytest.fixture()
def corpus_folder(tmpdir, sample_corpus, sub_folder):
    sample_corpus.to_files(tmpdir, sub_folder_name=sub_folder)
    return Path(tmpdir) / convert_to_path_name(sample_corpus.name)


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


def test_PieceOfWork__to_dict(caesar_quote):
    actual_dict = caesar_quote.to_dict()
    expected_dict = {
        "author": caesar_quote.author,
        "title": caesar_quote.title,
        "text": caesar_quote.text,
    }
    assert actual_dict == expected_dict
    assert json.loads(json.dumps(actual_dict)) == actual_dict


def test_PieceOfWork__to_file(caesar_quote, tmpdir):
    caesar_quote.to_file(tmpdir)

    expected_path = tmpdir / convert_to_path_name(caesar_quote.title) + ".json"
    assert expected_path.exists()

    with expected_path.open("r") as fh:
        file_content = json.load(fh)

    assert file_content == caesar_quote.to_dict()


def test_PieceOfWork__self_equality(caesar_quote):
    assert caesar_quote == caesar_quote


def test_PieceOfWork__equality_with_meta_data_change(caesar_quote):
    arrogant_quote = PieceOfWork(
        author="Caesar",
        title="arrogant quote",
        text=caesar_quote.text,
    )
    assert caesar_quote.author != arrogant_quote.author
    assert caesar_quote.title != arrogant_quote.title
    assert caesar_quote == arrogant_quote


def test_PieceOfWork__not_equal(caesar_quote):
    changed_quote = PieceOfWork(
        author=caesar_quote.author,
        title=caesar_quote.title,
        text=caesar_quote.text.replace(".", "!"),
    )
    assert caesar_quote != changed_quote


def test_PieceOfWork__no_comparison_to_other_types(caesar_quote):
    expected_error_message = "Cannot compare instances of types "
    expected_error_message += f"{type(caesar_quote)} and {type(caesar_quote.text)}."
    with pytest.raises(TypeError, match=expected_error_message):
        caesar_quote == caesar_quote.text
    with pytest.raises(TypeError, match=expected_error_message):
        caesar_quote.text == caesar_quote


def test_Corpus(caesar_quote, cato_quote, de_bello_gallico):
    name = "test corpus"
    corpus = Corpus(name)
    assert corpus.name == name
    assert len(corpus.works) == 0

    corpus.add_piece_of_work(caesar_quote)
    assert corpus.works == set([caesar_quote.title])
    assert corpus.authors == set([caesar_quote.author])

    corpus.add_piece_of_work(cato_quote)
    assert corpus.works == set([caesar_quote.title, cato_quote.title])
    assert corpus.authors == set([caesar_quote.author, cato_quote.author])

    corpus.add_piece_of_work(de_bello_gallico)
    assert corpus.works == set(
        [caesar_quote.title, cato_quote.title, de_bello_gallico.title]
    )
    assert caesar_quote.author == de_bello_gallico.author
    assert corpus.authors == set([caesar_quote.author, cato_quote.author])


def test_Corpus__no_duplicates(caesar_quote):
    name = "test corpus"
    corpus = Corpus(name)
    assert corpus.name == name
    assert len(corpus.works) == 0

    corpus.add_piece_of_work(caesar_quote)
    assert corpus.works == set([caesar_quote.title])

    arrogant_quote = PieceOfWork(
        author=caesar_quote.author, title="arrogant quote", text=caesar_quote.text
    )
    expected_message = f"Added text identical to pre-existing text {caesar_quote.title}"
    with pytest.raises(ValueError, match=expected_message):
        corpus.add_piece_of_work(arrogant_quote)


def test_Corpus__to_files(sample_corpus, tmpdir, sub_folder):
    sample_corpus.to_files(tmpdir, sub_folder_name=sub_folder)

    root_path = Path(tmpdir) / convert_to_path_name(sample_corpus.name)
    assert root_path.exists()

    for author in sample_corpus.authors:
        works = sample_corpus.get_works_from_author(author)
        for work in works:
            work_path = root_path / convert_to_path_name(author)
            work_path /= "" if sub_folder is None else sub_folder
            work_path /= convert_to_path_name(work.title) + ".json"
            assert work_path.exists()
            with work_path.open("r") as fh:
                content = json.load(fh)
            assert content == work.to_dict()


def test_Corpus__add_data_from_files(corpus_folder, sample_corpus):
    extracted_corpus = Corpus(name="extracted")
    extracted_corpus.add_data_from_files(corpus_folder, meta_keys=["author", "title"])
    assert len(extracted_corpus.works) == len(sample_corpus.works)
    assert len(extracted_corpus.authors) == len(sample_corpus.authors)
    assert extracted_corpus.hashes == sample_corpus.hashes


def test_Corpus__add_data_from_files__filter(corpus_folder, sample_corpus):
    extracted_corpus = Corpus(name="extracted")
    extracted_corpus.add_data_from_files(
        corpus_folder, filename_contains="_quot", meta_keys=["author", "title"]
    )
    assert len(extracted_corpus.works) < len(sample_corpus.works)
    assert len(extracted_corpus.authors) <= len(sample_corpus.authors)
    assert extracted_corpus.hashes.issubset(sample_corpus.hashes)


def test_convert_to_path_name():
    name = "Gaius Julius Ceasar"
    converted = convert_to_path_name(name)
    for char in string.whitespace:
        assert char not in converted
    assert name.split() == converted.split("_")
