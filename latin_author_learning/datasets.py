import json
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

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
    elif data is None:
        return ""
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

    @property
    def hashes(self) -> Set[str]:
        """
        Get all hashes of the works contained in the corpus.

        Returns
        -------
        Set[str]
            Set of hexdigests of the text hashes.
        """
        return set([w.hash.hexdigest() for w in self._works])

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

        Raises
        ------
        ValueError
            If specified author is not found in corpus.
        """
        if author not in self.authors:
            raise ValueError(
                f"Author '{author}' is not present in corpus '{self.name}'."
            )
        return [w for w in self._works if w.author == author]

    def to_files(self, path: Path, sub_folder_name: Optional[str] = None):
        """
        Write corpus to the file system.

        The file structure in which the content of the corpus
        is saved looks like this.
        ```
        corpus root path
        |
        |-> author paths
            |-> author sub path with dir name specified in `sub_folder_name`
                |-> file path corresponding to work title with `.json` suffix.
        ```

        Parameters
        ----------
        path : pathlib.Path
            Path in which the corpus should be stored.
            A root dir for the corpus files will be created within this directory.
        sub_folder_name : Optional[str]
            Name for sub-folders within the author directories. If None, fildes for
            the text are directly placed in the author directories.
        """
        root_path = path / convert_to_path_name(self.name)
        root_path.mkdir()

        for author in self.authors:
            author_path = root_path / convert_to_path_name(author)
            author_path.mkdir()
            if sub_folder_name is None:
                author_sub_path = author_path
            else:
                author_sub_path = author_path / sub_folder_name
                author_sub_path.mkdir()

            works = self.get_works_from_author(author)
            for work in works:
                work.to_file(author_sub_path)

    def _add_works(
        self,
        works_path: Iterable[Path],
        author: str,
        filename_contains: str,
        **kwargs: Any,
    ):
        filtered_paths = filter(lambda p: p.suffix.lower() == ".json", works_path)
        filtered_paths = filter(lambda p: filename_contains in p.name, filtered_paths)
        for file in filtered_paths:
            text = read_text(file, **kwargs)
            work = PieceOfWork(author=author, text=text, title=file.name)
            self.add_piece_of_work(work)

    def add_data_from_files(
        self,
        corpus_root_path: Path,
        filename_contains: str = "",
        **kwargs: Any,
    ):
        """
        Add data from structured JSON files.

        These files are organized in the the following folder structure.
        ```
        corpus root path
        |
        |-> author paths
            |-> sub path
            |   |-> file path corresponding to work title with `.json` suffix.
            |
            |-> file path corresponding to work title with `.json` suffix.
        ```
        The individual files are read via `latin_author_learning.datasets.read_text`.

        Parameters
        ----------
        corpus_root_path : pathlib.Path
            Path where the author folders are contained.
        filename_contains : str
            Only read files with file names that contain this string. By default,
            there is no further filtering of JSON files.
        **kwargs : Any
            Keyword arguments to pass on to `latin_author_learning.datasets.read_text`.
        """
        author_paths = corpus_root_path.iterdir()
        for author in filter(lambda p: p.is_dir(), author_paths):
            for sub_path in author.iterdir():
                if sub_path.is_dir():
                    files: Iterable[Path] = sub_path.iterdir()
                else:
                    files = [sub_path]
                self._add_works(files, author.name, filename_contains, **kwargs)


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
