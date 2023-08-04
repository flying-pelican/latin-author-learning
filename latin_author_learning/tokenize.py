from typing import List

from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer

STARTS = "[STA]"
ENDS = "[END]"
WORD_SEPARATOR = " "


def get_subtoken_strings(tokenizer_path: str) -> List[str]:
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
    sentence_tokenizer = SentenceTokenizer()
    word_tokenizer = WordTokenizer()

    sentences = sentence_tokenizer.tokenize(text)
    tokenized_sentences = map(
        lambda sentence: word_tokenizer.tokenize(sentence), sentences
    )
    tokenized_sentences = map(
        lambda sentence: [STARTS] + sentence + [ENDS], tokenized_sentences
    )

    sentences_with_meta_info = map(
        lambda sentence: WORD_SEPARATOR.join(sentence), tokenized_sentences
    )
    text_with_meta_info = WORD_SEPARATOR.join(sentences_with_meta_info)
    return text_with_meta_info
