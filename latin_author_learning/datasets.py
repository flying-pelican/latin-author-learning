import json
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

SECTION_SEPARATOR = "\n"


def _add_text_with_separator(text: str, added_text: str) -> str:
    if text and added_text:
        text += SECTION_SEPARATOR
    text += added_text
    return text


def extract_text(
    data: Union[str, List[Union[str, List, Dict]], Dict[Any, Union[str, List, Dict]]],
    meta_keys: Optional[List[str]] = None,
    meta_key_prefix: Optional[str] = None,
) -> str:
    """
    Extract text from a nested data structure.

    Text is concatenated for all keys in that file that are not specified as meta
    data keys. This holds irrespective of the nesting level.

    Parameters
    ----------
    data : Union[str, List[Union[str, List, Dict]], Dict[Any, Union[str, List, Dict]]]
        Nested data structure. All values in the dicts within this data structure either
        have to be strings, lists, or dicts.

    meta_keys : Optional[List[str]]
        Keys to be excluded from the text extraction. These keys are excluded
        irrespective of the nesting level.

    meta_key_prefix : Optional[str]
        Prefix for meta keys. All keys starting with that prefix will be automatically
        considered as meta keys.

    Returns
    -------
    str
        Extracted text.

    Raises
    ------
    TypeError
        If a key within data is neither a string, nor a list, nor a dict.
    """
    if meta_keys is None:
        meta_keys = []
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        text = ""
        for k in data:
            key_has_meta_prefix = meta_key_prefix is not None and k.startswith(
                meta_key_prefix
            )
            if k not in meta_keys and not key_has_meta_prefix:
                additional_text = extract_text(data[k], meta_keys, meta_key_prefix)
                text = _add_text_with_separator(text, additional_text)
        return text
    elif isinstance(data, list):
        text = ""
        for element in data:
            additional_text = extract_text(element, meta_keys, meta_key_prefix)
            text = _add_text_with_separator(text, additional_text)
        return text
    else:
        actual_type = type(data)
        raise TypeError(
            f"""
                Got unexpected type {actual_type} from parsed json.
                All values are assumed to be strings, or lists, or dicts.
                """
        )


def read_text(
    file_path: Path,
    meta_keys: Optional[List[str]] = None,
    meta_key_prefix: Optional[str] = None,
) -> str:
    """
    Read text from a nested file using `latin_author_learning.datasets.extract_text`.

    Currently only JSON format is supported. This could be extended to XML later.

    Parameters
    ----------
    file_path : pathlib.Path
        Nested data file from which the text is extracted.

    meta_keys : Optional[List[str]]
        Keys to be excluded from the text extraction. These keys are excluded
        irrespective of the nesting level.

    meta_key_prefix : Optional[str]
        Prefix for meta keys. All keys starting with that prefix will be automatically
        considered as meta keys.

    Returns
    -------
    str
        Extracted text.

    Raises
    ------
    NotImplementedError
        If the file name under `file_path` does not a have a json file name extension.
    """
    if file_path.suffix.lower() != ".json":
        raise NotImplementedError(
            f"""
            Wrong file format of file {file_path}.
            Currently only JSON files are supported.
            """
        )
    with file_path.open("r") as fh:
        data = json.load(fh)
    return extract_text(data, meta_keys, meta_key_prefix)


class PieceOfWork(object):
    """
    Class for a piece of work.

    An MD5 hash for this piece of work is created upon initialization so that the
    identity of texts can be easily verified.

    Parameters
    ----------
    author : str
        Author of the given piece of work.
    text : str
        Actual text (payload data).
    title : str
        Human readable name for the piece of work.
    """

    def __init__(self, author: str, text: str, title: str):
        self.author = author
        self.text = text
        self.title = title
        self.hash = md5(text.encode(), usedforsecurity=False)

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to a dict that can be serialized as json.

        Returns
        -------
        Dict[str, str]
            Payload and meta data as a dict with the following keys.
            `author`, `title`, and `text`.
        """
        return {
            "author": self.author,
            "title": self.title,
            "text": self.text,
        }

    def to_file(self, path: Path):
        """
        Dump payload and meta data as a json.

        Parameters
        ----------
        path : pathlib.Path
            Directory where data should be dumped.
        """
        file_content = self.to_dict()
        file_name = convert_to_path_name(self.title) + ".json"
        file_path = path / file_name
        with file_path.open("w") as fh:
            json.dump(file_content, fh)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot compare instances of types {self.__class__} and {type(other)}."
            )
        return self.hash.hexdigest() == other.hash.hexdigest()


class Corpus(object):
    """
    Class for a corpus with several piece of work.

    Parameters
    ----------
    name : str
        Name of the corpus.
    """

    def __init__(self, name: str):
        self.name = name
        self._works: List[PieceOfWork] = []

    @property
    def works(self) -> Set[str]:
        """
        Get works contained in the corpus.

        Returns
        -------
        Set[str]
            Set of the titles of all works contained in the corpus.
        """
        return set([w.title for w in self._works])

    @property
    def authors(self) -> Set[str]:
        """
        Get all authors of the works contained in the corpus.

        Returns
        -------
        Set[str]
            Set of the author names.
        """
        return set([w.author for w in self._works])

    def _get_pre_existing_work(self, new_work: PieceOfWork) -> Optional[PieceOfWork]:
        for pre_existing_work in self._works:
            if pre_existing_work == new_work:
                return pre_existing_work
        return None

    def add_piece_of_work(self, new_work: PieceOfWork):
        """
        Append a piece of work to the corpus.

        Checks that no duplicates are added using MD5 hashes as checksums.

        Parameters
        ----------
        new_work : PieceOfWork
            New piece of work to be added.
        """
        pre_existing_work = self._get_pre_existing_work(new_work)
        if pre_existing_work is not None:
            raise ValueError(
                f"Added text identical to pre-existing text {pre_existing_work.title}"
            )
        self._works.append(new_work)

    def get_works_from_author(self, author: str) -> List[PieceOfWork]:
        """
        Get works from a particular author.

        Parameters
        ----------
        author : str
            Author name to be matched with the works in the corpus.

        Returns
        -------
        List[PieceOfWork]
            Works written by the specified author.
        """
        return [w for w in self._works if w.author == author]

    def to_files(self, path: Path):
        """
        Write corpus to the file system.

        The file structure in which the content of the corpus
        is saved looks like this.
        ```
        corpus root path
        |
        |-> author paths
            |-> author sub path with dir name `opensource`
                |-> file path corresponding to work title with `.json` suffix.
        ```

        Parameters
        ----------
        path : pathlib.Path
            Path in which the corpus should be stored.
            A root dir for the corpus files will be created within this directory.
        """
        root_path = path / convert_to_path_name(self.name)
        root_path.mkdir()

        for author in self.authors:
            author_path = root_path / convert_to_path_name(author)
            author_path.mkdir()
            author_sub_path = author_path / "opensource"
            author_sub_path.mkdir()

            works = self.get_works_from_author(author)
            for work in works:
                work.to_file(author_sub_path)


def convert_to_path_name(name: str) -> str:
    """
    Convert arbitrary string to a suitable file name.

    Parameters
    ----------
    name : str
        Name to be converted.

    Returns
    -------
    str
        Converted name.
    """
    return "_".join(name.split())
