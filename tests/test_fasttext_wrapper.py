from pathlib import Path
import pytest

import numpy as np
import fasttext

from latin_author_learning.corpus import Corpus, PieceOfWork
from latin_author_learning.fasttext_wrapper import DatasetWrapper, _works_as_str


@pytest.fixture(scope="session")
def caesar_tacitus_corpus():
    corpus_name = "test_Cicero_Tacitus"
    file_dir = Path(__file__).parent
    path = file_dir / "data/small_sample_corpus"
    corpus = Corpus("test_Cicero_Tacitus")
    corpus.add_data_from_files(path / corpus_name)
    return corpus


@pytest.fixture()
def training_file(tmpdir):
    return tmpdir / "training_data"


@pytest.fixture(scope="session")
def sample_works():
    bellum_gallicum = PieceOfWork(
        title="De bello gallico",
        author="Caesar",
        text="Gallia omnis divisa est in partes tres",
    )
    caesar_quote = PieceOfWork(title="Quote", author="Caesar", text="Veni, vidi, vici.")
    return [bellum_gallicum, caesar_quote]


@pytest.mark.parametrize("include_labels", [True, False])
def test__works_as_str__multiple_works(sample_works, include_labels):
    stringyfied_works = _works_as_str(sample_works, include_labels)
    works_from_string = stringyfied_works.split("\n")
    for index, work in enumerate(sample_works):
        assert works_from_string[index] == _works_as_str([work], include_labels)


def test__works_as_str__author_encoding_train(sample_works):
    for work in sample_works:
        modified_author_name = work.author.lower().replace(" ", "_")
        expected_label = f"__label__{modified_author_name}"
        assert expected_label in _works_as_str([work], include_labels=True)


def test__works_as_str__no_author_info_test(sample_works):
    for work in sample_works:
        assert work.text == _works_as_str([work], include_labels=False)


@pytest.mark.parametrize("include_labels", [True, False])
def test__works_as_str__text(sample_works, include_labels):
    for work in sample_works:
        assert work.text in _works_as_str([work], include_labels)


@pytest.mark.parametrize("include_labels", [True, False])
def test__works_as_str__newlines(sample_works, include_labels):
    for work in sample_works:
        work.text = work.text.replace(" ", "\n")
        assert "\n" not in _works_as_str([work], include_labels)


def test_DatasetWrapper__train_predict(caesar_tacitus_corpus, training_file):
    dataset_wrapper = DatasetWrapper(caesar_tacitus_corpus)
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


@pytest.mark.parametrize("fraction_for_test", [0.25, 0.5, 0.75])
def test_DatasetWrapper__get_train_data(
    caesar_tacitus_corpus, fraction_for_test, training_file
):
    dataset_wrapper = DatasetWrapper(caesar_tacitus_corpus, fraction_for_test)
    dataset_wrapper.get_training_data(training_file)

    with open(training_file, "r") as f:
        dumped_training_data = f.read()
    training_lines = dumped_training_data.split("\n")
    assert len(training_lines) == len(dataset_wrapper.train_works)
    for index, work in enumerate(dataset_wrapper.train_works):
        assert " ".join(work.text.split()) in training_lines[index]


@pytest.mark.parametrize("fraction_for_test", [0.25, 0.5, 0.75])
def test_DatasetWrapper__get_test_data(caesar_tacitus_corpus, fraction_for_test):
    dataset_wrapper = DatasetWrapper(caesar_tacitus_corpus, fraction_for_test)
    test_data = dataset_wrapper.get_test_data()
    test_lines = test_data.split("\n")
    assert len(test_lines) == len(dataset_wrapper.test_works)
    for index, work in enumerate(dataset_wrapper.test_works):
        assert " ".join(work.text.split()) in test_lines[index]


@pytest.mark.parametrize("fraction_for_test", [0.25, 0.5, 0.75])
def test_DatasetWrapper__fraction_for_test(caesar_tacitus_corpus, fraction_for_test):
    dataset_wrapper = DatasetWrapper(caesar_tacitus_corpus, fraction_for_test)
    assert len(dataset_wrapper.test_works) == fraction_for_test * len(
        caesar_tacitus_corpus.works
    )
    assert len(dataset_wrapper.train_works) == (1.0 - fraction_for_test) * len(
        caesar_tacitus_corpus.works
    )

    train_corpus = Corpus("train")
    train_corpus.add_list_of_works(dataset_wrapper.train_works)
    test_corpus = Corpus("test")
    test_corpus.add_list_of_works(dataset_wrapper.test_works)
    assert test_corpus.hashes.intersection(train_corpus.hashes) == set()


@pytest.mark.parametrize("fraction_for_test", [np.nan, -1, -1.5, 1.1, -0.01])
def test_DatasetWrapper__invalid_fraction_for_test(
    caesar_tacitus_corpus, fraction_for_test
):
    with pytest.raises(ValueError):
        _ = DatasetWrapper(caesar_tacitus_corpus, fraction_for_test)
