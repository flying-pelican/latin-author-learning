import json
from pathlib import Path
from typing import Any, Dict, List, Union
from uuid import uuid4

SECTION_SEPARATOR = "\n"


def extract_text(
    data: Union[str, Dict[Any, Union[str, Dict]]], meta_data_keys: List[str]
) -> str:
    """
    Extracts text from a nested data structure. Text is concatenated for all keys,
    that are not specified as meta data keys.

    Parameters
    ----------
    data : Union[str, Dict[Any, Union[str, Dict]]]
        Nested data structure. All values in the dicts within this data structure either
        have to be strings or dicts.

    meta_data_keys : List[str]
        Keys to excluded from the text extraction. These keys are excluded irrespective
        of the nesting level.

    Returns
    -------
    str
        Extracted text

    Raises
    ------
    TypeError
        If a key within data is neither a string nor a dict.
    """
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        text = ""
        for k in data:
            if k not in meta_data_keys:
                additional_text = extract_text(data[k], meta_data_keys)
                if text and additional_text:
                    text += SECTION_SEPARATOR
                text += additional_text
        return text
    else:
        raise TypeError(
            """
                Got unexpected type from parsed json.
                All values are assumed to be strings.
                """
        )


def read_text(file_path: Path, meta_data_keys: List[str]) -> str:
    """
    Reads text from a nested file using `latin_author_learning.datasets.extract_text`.
    Currently only JSON format is supported.

    Parameters
    ----------
    file_path : pathlib.Path
        Nested data file from which the text is extracted.

    meta_data_keys : List[str]
        Keys to excluded from the text extraction. These keys are excluded irrespective
        of the nesting level.

    Returns
    -------
    str
        Extracted text

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
    return extract_text(data, meta_data_keys)


class PieceOfWork(object):
    """
    Class for a piece of work. A UUID for this piece of work
    is set up when an instance of this class is created.
    """

    def __init__(self, author: str, text: str, name: str):
        self.author = author
        self.text = text
        self.name = name
        self.uuid = uuid4()
