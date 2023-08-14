import json
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
