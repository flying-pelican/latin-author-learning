import math
from pathlib import Path
from typing import List

from fasttext.FastText import _FastText

from latin_author_learning.corpus import Corpus, PieceOfWork
from latin_author_learning.tokenize import WORD_SEPARATOR, convert_to_tokens


def _text_to_chunks(text: str, chunksize: int) -> List[str]:
    words = text.split(WORD_SEPARATOR)
    chunks = []
    start_chunk = 0
    while start_chunk < len(words):
        end_chunk = start_chunk + chunksize
        if end_chunk < len(words):
            chunk = words[start_chunk:end_chunk]
        else:
            chunk = words[start_chunk:]
        chunks.append(WORD_SEPARATOR.join(chunk))
        start_chunk = end_chunk
    return chunks


def _works_as_str(
    works: List[PieceOfWork], include_labels: bool, chunksize: int
) -> str:
    result = []
    for work in works:
        text = convert_to_tokens(work.text)
        author = "_".join(work.author.lower().split())
        chunks = _text_to_chunks(text, chunksize)
        for chunk in chunks:
            if include_labels:
                result.append(f"__label__{author} {chunk}")
            else:
                result.append(chunk)
    return "\n".join(result)


class DatasetWrapper(object):
    """
    Makes data from a Corpus compatible with the methods of fasttext.

    Parameters
    ----------
    corpus : Corpus
        Corpus from which train and test data chall be taken from.
    fraction_for_test : float
        Fraction of sample text that will end up in the test sample.
        Must be between 0.0 and 1.0.
    chunksize : int
        Maximal number of words in a chunk. Each junk is considered
        as an independet data point.

    Raises
    ------
    ValueError
        If `fraction_for_test` <= 0.0 or >= 1.0.
    """

    def __init__(
        self, corpus: Corpus, fraction_for_test: float = 0.25, chunksize: int = 500
    ):
        self._corpus = corpus
        if fraction_for_test < 1.0 and fraction_for_test > 0.0:
            self.fraction_for_test = fraction_for_test
        else:
            raise ValueError(
                "Parameter `fraction_for_test` is not between 0.0 and 1.0."
            )
        self.chunksize = chunksize
        self._split_train_test()

    def get_training_data(self, file: Path) -> None:
        """
        Dump training data to the specified file.

        Parameters
        ----------
        file : pathlib.Path
            Filepath to which the training data should be dumped to.
        """
        training_string = _works_as_str(
            self.train_works, include_labels=True, chunksize=self.chunksize
        )
        self._text_to_file(training_string, file)

    def get_validation_data(self, file: Path) -> None:
        """
        Dump validation data to the specified file.

        Parameters
        ----------
        file : pathlib.Path
            Filepath to which the validation data should be dumped to.
        """
        validation_string = _works_as_str(
            self.test_works, include_labels=True, chunksize=self.chunksize
        )
        self._text_to_file(validation_string, file)

    def get_test_data(self) -> str:
        """
        Obtain test data as a string.

        Returns
        -------
        str
            Test data as a single string. Each line corresponds to a single
            piece of work.
        """
        return _works_as_str(
            self.test_works, include_labels=False, chunksize=self.chunksize
        )

    def _split_train_test(self):
        self.train_works = []
        self.test_works = []
        for author in self._corpus.authors:
            author_works = self._corpus.get_works_from_author(author)
            num_works_in_train = math.floor(
                len(author_works) * (1.0 - self.fraction_for_test)
            )
            self.train_works += author_works[:num_works_in_train]
            self.test_works += author_works[num_works_in_train:]

    @staticmethod
    def _text_to_file(text: str, file: Path) -> None:
        with open(file, "w") as f:
            f.write(text)


def model_to_vec_str(model: _FastText) -> str:
    """
    Convert a fasttext model into the contens of a vec. file.

    Parameters
    ----------
    model : fasttext.FastText._FastText
        Model as returned by `fasttext.train_unsupervised`.

    Returns
    -------
    str
        Contents of the respective .vec file.
    """
    words = model.get_words()
    vec_str = f"{len(words)} {model.get_dimension()}"
    for word in words:
        word_vec = model.get_word_vector(word)
        word_vec_str = " ".join([str(num) for num in word_vec])
        vec_str += f"\n{word} {word_vec_str}"
    return vec_str
