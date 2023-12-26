from pathlib import Path
import math
import pytest

import numpy as np
import fasttext

from latin_author_learning.corpus import Corpus, PieceOfWork
from latin_author_learning.tokenize import convert_to_tokens
from latin_author_learning.fasttext_wrapper import (
    _text_to_chunks,
    DatasetWrapper,
    model_to_vec_str,
    _works_as_str,
)


@pytest.fixture(scope="session")
def cicero_tacitus_corpus():
    corpus_name = "test_Cicero_Tacitus"
    file_dir = Path(__file__).parent
    path = file_dir / "data/small_sample_corpus"
    corpus = Corpus("test_Cicero_Tacitus")
    corpus.add_data_from_files(
        path / corpus_name,
        meta_keys=["fileDesc", "teiHeader"],
        meta_key_prefix="@",
    )
    return corpus


@pytest.fixture()
def training_file(tmpdir):
    return tmpdir / "training_data.txt"


@pytest.fixture()
def valid_file(tmpdir):
    return tmpdir / "test_data.txt"


@pytest.fixture(scope="session")
def sample_works():
    bellum_gallicum = PieceOfWork(
        title="De bello gallico",
        author="Caesar",
        text="Gallia omnis divisa est in partes tres",
    )
    caesar_quote = PieceOfWork(title="Quote", author="Caesar", text="Veni, vidi, vici.")
    return [bellum_gallicum, caesar_quote]


@pytest.fixture()
def embedding_model(cicero_tacitus_corpus, training_file):
    dataset_wrapper = DatasetWrapper(cicero_tacitus_corpus)
    dataset_wrapper.get_training_data(training_file)
    model = fasttext.train_unsupervised(str(training_file))
    return model


def assert_file_contents_in_works(file, works, chunksize):
    with open(file, "r") as f:
        dumped_data = f.read()
    lines = dumped_data.split("\n")
    assert len(lines) >= len(works)
    all_text = " ".join([w.text for w in works])
    for line in lines:
        content = " ".join([w for w in line.split() if not w.startswith("__label__")])
        assert content in convert_to_tokens(all_text)
        assert len(content.split()) <= chunksize


@pytest.mark.parametrize("chunksize", [5, 7, 17, 500])
def test__text_to_chunks(cicero_tacitus_corpus, chunksize):
    work = cicero_tacitus_corpus._works[0]
    text = convert_to_tokens(work.text)

    chunks = _text_to_chunks(text, chunksize)

    assert " ".join(chunks) == text
    assert len(chunks) == math.ceil(len(text.split(" ")) / chunksize)
    for chunk in chunks:
        assert len(chunk.split()) <= chunksize


@pytest.mark.parametrize("include_labels", [True, False])
@pytest.mark.parametrize("chunksize", [5, 17, 500])
def test__works_as_str__multiple_works(sample_works, include_labels, chunksize):
    all_works = _works_as_str(sample_works, include_labels, chunksize)
    individual_works = [
        _works_as_str([w], include_labels, chunksize) for w in sample_works
    ]
    assert all_works == "\n".join(individual_works)


def test__works_as_str__author_encoding_train(sample_works):
    for work in sample_works:
        modified_author_name = work.author.lower().replace(" ", "_")
        expected_label = f"__label__{modified_author_name}"
        work_str = _works_as_str([work], include_labels=True, chunksize=15)
        assert expected_label in work_str
        assert work_str.count(expected_label) == work_str.count("__label__")
        assert work_str.count(expected_label) == len(work_str.split("\n"))


def test__works_as_str__no_author_info_test(sample_works):
    for work in sample_works:
        assert "__label__" not in _works_as_str(
            [work], include_labels=False, chunksize=15
        )


@pytest.mark.parametrize("include_labels", [True, False])
@pytest.mark.parametrize("chunksize", [5, 17, 500])
def test__works_as_str__text(sample_works, include_labels, chunksize):
    for work in sample_works:
        stringified_work = _works_as_str([work], include_labels, chunksize)
        for chunk in stringified_work.split("\n"):
            chunk_without_label = " ".join(
                [w for w in chunk.split() if not w.startswith("__label__")]
            )
            assert chunk_without_label in convert_to_tokens(work.text)


@pytest.mark.parametrize("include_labels", [True, False])
@pytest.mark.parametrize("chunksize", [5, 17, 500])
def test__works_as_str__chunksize(sample_works, include_labels, chunksize):
    lines = _works_as_str(sample_works, include_labels, chunksize).split("\n")
    for line in lines:
        assert len(line.split()) <= chunksize + line.count("__label__")


def test_DatasetWrapper__train_predict(cicero_tacitus_corpus, training_file):
    dataset_wrapper = DatasetWrapper(cicero_tacitus_corpus)
    dataset_wrapper.get_training_data(training_file)
    model = fasttext.train_supervised(input=str(training_file))

    test_data = dataset_wrapper.get_test_data()
    for text in test_data.split("\n"):
        prediction = model.predict(text)
        assert len(prediction) == 2
        assert len(prediction[0]) == 1
        assert prediction[0][0].startswith("__label__")
        assert len(prediction[1]) == 1
        assert prediction[1][0] > 0.0
        assert prediction[1][0] < 1.0


def test_DatasetWrapper__train_validate(
    cicero_tacitus_corpus, training_file, valid_file
):
    dataset_wrapper = DatasetWrapper(cicero_tacitus_corpus)
    dataset_wrapper.get_training_data(training_file)
    dataset_wrapper.get_validation_data(valid_file)
    model = fasttext.train_supervised(input=str(training_file))

    validation_results = model.test(str(valid_file))
    assert len(validation_results) == 3
    assert validation_results[0] == len(dataset_wrapper.test_works)


@pytest.mark.parametrize("fraction_for_test", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("chunksize", [5, 17, 47, 500])
def test_DatasetWrapper__get_train_data(
    cicero_tacitus_corpus, training_file, fraction_for_test, chunksize
):
    dataset_wrapper = DatasetWrapper(
        cicero_tacitus_corpus, fraction_for_test, chunksize
    )
    dataset_wrapper.get_training_data(training_file)
    assert_file_contents_in_works(training_file, dataset_wrapper.train_works, chunksize)


@pytest.mark.parametrize("fraction_for_test", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("chunksize", [5, 17, 47, 500])
def test_DatasetWrapper__get_validation_data(
    cicero_tacitus_corpus, valid_file, fraction_for_test, chunksize
):
    dataset_wrapper = DatasetWrapper(
        cicero_tacitus_corpus, fraction_for_test, chunksize
    )
    dataset_wrapper.get_validation_data(valid_file)
    assert_file_contents_in_works(valid_file, dataset_wrapper.test_works, chunksize)


@pytest.mark.parametrize("fraction_for_test", [0.25, 0.5, 0.75])
def test_DatasetWrapper__get_test_data(cicero_tacitus_corpus, fraction_for_test):
    dataset_wrapper = DatasetWrapper(cicero_tacitus_corpus, fraction_for_test)
    test_data = dataset_wrapper.get_test_data()
    test_lines = test_data.split("\n")
    assert len(test_lines) == len(dataset_wrapper.test_works)
    for index, work in enumerate(dataset_wrapper.test_works):
        assert convert_to_tokens(work.text) in test_lines[index]


@pytest.mark.parametrize("fraction_for_test", [0.25, 0.5, 0.75])
def test_DatasetWrapper__fraction_for_test(cicero_tacitus_corpus, fraction_for_test):
    dataset_wrapper = DatasetWrapper(cicero_tacitus_corpus, fraction_for_test)
    assert len(dataset_wrapper.test_works) == fraction_for_test * len(
        cicero_tacitus_corpus.works
    )
    assert len(dataset_wrapper.train_works) == (1.0 - fraction_for_test) * len(
        cicero_tacitus_corpus.works
    )

    train_corpus = Corpus("train")
    train_corpus.add_list_of_works(dataset_wrapper.train_works)
    test_corpus = Corpus("test")
    test_corpus.add_list_of_works(dataset_wrapper.test_works)
    assert test_corpus.hashes.intersection(train_corpus.hashes) == set()


@pytest.mark.parametrize("fraction_for_test", [np.nan, -1, -1.5, 1.1, -0.01])
def test_DatasetWrapper__invalid_fraction_for_test(
    cicero_tacitus_corpus, fraction_for_test
):
    with pytest.raises(ValueError):
        _ = DatasetWrapper(cicero_tacitus_corpus, fraction_for_test)


def test_model_to_vec_str(embedding_model):
    vec_str = model_to_vec_str(embedding_model)
    words = embedding_model.get_words()

    vec_lines = vec_str.split("\n")
    assert len(vec_lines) == len(words) + 1

    header_info = vec_lines[0].split()
    assert len(header_info) == 2
    assert int(header_info[0]) == len(words)
    assert int(header_info[1]) == embedding_model.get_dimension()

    for line in vec_lines[1:]:
        line_contents = line.split()
        assert line_contents[0] in words
        embedding_vec = np.array(line_contents[1:], dtype=float)
        assert len(embedding_vec) == embedding_model.get_dimension()
        assert np.sum(embedding_vec**2) > 0.0
