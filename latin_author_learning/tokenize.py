from pathlib import Path
from typing import List

from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer

WORD_SEPARATOR = " "
DELIMITERS = [".", ",", ";", "?", "!", ":"]


def get_subtoken_strings(tokenizer_path: Path) -> List[str]:
    """
    Load tokens for a subword encoder from a file.

    Parameters
    ----------
    tokenizer_path : pathlib.Path
        Path for the vocabulary file containing the tokens. The tokens must be
        separated by newlines and are expected to be quoted.

    Returns
    -------
    List[str]
        List with tokens are they are found in the file. If the tokens have
        trailing underscores, these are kept.
    """
    subtoken_strings = []
    with open(tokenizer_path, "r") as f:
        for line in f:
            s = line.strip()
            # Some vocab files wrap words in single quotes, but others don't
            if (s.startswith("'") and s.endswith("'")) or (
                s.startswith('"') and s.endswith('"')
            ):
                s = s[1:-1]
            subtoken_strings.append(s)
    return subtoken_strings


def _tokenize_words(sentence: str, word_tokenizer: WordTokenizer) -> str:
    words = word_tokenizer.tokenize(sentence)
    words = filter(lambda w: w not in DELIMITERS, words)
    sentence = WORD_SEPARATOR.join(words)
    return sentence


def convert_to_tokens(text: str) -> str:
    """
    Tokenize a raw text, using the tokenizers from `cltk.tokenizers.lat.lat`.

    The tokens returned are the words as lowercase strings. Sentence boundaries are
    not visible in the tokenized text as in classical Latin incriptions. Punctuation
    is ignored. A simple blank is used as a word delimiter.

    Parameters
    ----------

    text : str
        Raw text to be tokenized.

    Returns
    -------
    str
       Tokenized text.
    """
    sentence_tokenizer = SentenceTokenizer()
    word_tokenizer = WordTokenizer()

    sentences = sentence_tokenizer.tokenize(text)
    sentences = map(lambda sentence: sentence.lower(), sentences)
    sentences = map(
        lambda sentence: _tokenize_words(sentence, word_tokenizer), sentences
    )
    tokenized_text = "".join(sentences)

    return tokenized_text
