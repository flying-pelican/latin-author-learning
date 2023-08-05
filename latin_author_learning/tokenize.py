from pathlib import Path
from typing import List

from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from torchnlp.encoders.text import SubwordEncoder

STARTS = "STA"
ENDS = "END"
WORD_SEPARATOR = " "


def get_subtoken_strings(tokenizer_path: Path) -> List[str]:
    """
    Loads tokens for a subword encoder from a file.

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


def convert_to_tokens(text: str) -> str:
    """
    Tokenizes a raw text, using the word and sentence tokenizers of
    `cltk.tokenizers.lat.lat`.

    The tokens return are the words as lowercase strings. Sentence boundaries are
    indicated by control sequences in upper case. A simple blank is used as a word
    delimiter.

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
    tokenized_sentences = map(
        lambda sentence: word_tokenizer.tokenize(sentence), sentences
    )
    tokenized_sentences = map(
        lambda sentence: [STARTS] + sentence + [ENDS], tokenized_sentences
    )

    sentences_w_control_seqs = map(
        lambda sentence: WORD_SEPARATOR.join(sentence), tokenized_sentences
    )
    text_w_control_seqs = WORD_SEPARATOR.join(sentences_w_control_seqs)
    return text_w_control_seqs


class SentenceAwareEncoder(SubwordEncoder):
    """
    Derived class of `torchnlp.encoders.text.SubwordEncoder` that ensures a
    representation of the control sequences used in
    `latin_author_learning.tokenize.convert_to_tokens` as a single token.
    """

    control_sequences = [STARTS, ENDS]

    def __init__(self, vocabulary: List[str], *args, **kwargs):
        formatted_cs = [word + "_" for word in self.control_sequences]
        super().__init__(formatted_cs + vocabulary, *args, **kwargs)
