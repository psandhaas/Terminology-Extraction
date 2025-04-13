#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Fr, 14.02.25                                                     #
# ========================================================================== #

from bibtexparser.latexenc import latex_to_unicode
from collections import Counter
from glob import glob
import logging
from nltk.corpus import reuters
import numpy as np
import os
from os import PathLike
import pandas as pd
import psutil
import re
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from spacy.tokens import Doc, Span, Token
from string import punctuation
from time import time
from typing import Iterator, Literal, Tuple


logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%H:%M:%S"
)


invalid_punct_chars: str = "".join(
    [char for char in punctuation if char not in ["'", "-", "/"]]
)
patterns: dict[re.Pattern] = {
    "latex": re.compile(
        # match for \abc{...}, \abc[...]{...}, $...$, $$...$$
        r"\\|\[|\]|{|}|\$+"
    ),
    "xml_html": re.compile(
        # match for <...> or </...>
        r"</?|>"
    ),
    "invalid_punct": re.compile(
        rf"[{re.escape(invalid_punct_chars)}]", re.I
    ),
    "et_al": {
        "tok1": re.compile(r"^et$", re.I),
        "tok2": re.compile(r"^al\.?$", re.I)
    },
    "hyphenated_mw_pat": re.compile(
        r"([a-zA-Z]+(-[a-zA-Z]+(-[a-zA-Z]+)*)+)"
    ),
    "accent_contraction_pat": re.compile(
        r"(?<=[a-z])[´`’](?=[a-z])", re.I
    )
}


### DATA LOADING ###
def load_reuters() -> list[str]:
    """Load raw texts from `nltk.corpus.reuters`."""
    return [reuters.raw(fid) for fid in reuters.fileids()]


def latex2unicode(text: str) -> str:
    """Convert LaTeX special characters to unicode.

    E.g.: `"et t{\\'e}l{\\'e}chargeable"` -> `"et téléchargeable"`
    """
    return latex_to_unicode(rf"{text}")


def entry_fields(
    entry_m: re.Match, keys: list[str] = None
) -> dict[str, str] | None:
    """Extract fields as key-value pairs from a bibtex entry."""
    if entry_m is not None:
        entry = re.sub(r"[\{\}]", "", entry_m.group("entry"))
        fields = {}
        key: str = None
        value: str = None
        for ln in re.split(r"\n", entry):
            if re.match(r"^\s{6}", ln) is not None:  # continuation of value
                v = ln.lstrip()
                # decode latex formatting (e.g. {\'}e)
                v = latex2unicode(v)
                value = " ".join([value, v])
            elif re.match(r"^\s{4}", ln) is not None:  # key
                ln = re.sub(r"\s+", " ", ln)
                k, v = [
                    splt.strip() for splt in ln.split(" = ", 1)
                    if len(splt.strip()) > 0
                ]
                v = v.lstrip(r'"')  # remove left quotation mark
                v = v.rstrip(',')  # remove trailing line-comma
                v = v.rstrip('"')  # remove right quotation mark
                v = latex_to_unicode(rf"{v}")
                if key is not None:  # append last k-v
                    fields[key] = value
                key = k
                value = v
        if key is not None and value is not None:
            fields[key] = value
        if keys is not None:
            cols = [col for col in keys if col in fields.keys()]
            fields = {col: fields[col] for col in cols}
        return fields
    return None


def parse_bibtex_abstracts(
    bibtex: str | PathLike, english_only: bool = True,
    save_dir: str = None
) -> list[str]:
    """Parse abstracts in bibtex entries from a string or file path. Returns
    individual abstracts as unicode strings and optionally excludes
    non-english entries (default).

    Provide `save_dir` to save parsed entries to .txt-files.
    """
    entry_pat: re.Pattern = re.compile(
        r"(?:^|\}\n)(?P<entry>@.+?\n)(?=@|$)", re.M | re.S
    )
    try:  # read bibtex from file
        with open(bibtex, "r", encoding="utf-8") as f:
            bibtex = f.read()
    except FileNotFoundError:  # assume bibtex is a string
        pass

    entries: list[str] = []
    logger.info("Parsing bibtex entries...")
    t0 = time()
    for entry_m in entry_pat.finditer(bibtex):
        if (entry := entry_fields(
            entry_m, keys=["abstract", "language"]
        )) is not None:
            if (abstract := entry.get("abstract", None)) is not None:
                abstract = latex2unicode(abstract)
                if english_only:
                    # ~94% of abstracts have no language & ~2% are "eng"
                    if entry.get("language", None) in ["eng", None]:
                        entries.append(abstract)
                    else:
                        continue
                else:
                    entries.append(abstract)
            else:
                continue

    m, s = [int(t) for t in divmod(time() - t0, 60)]
    logger.info(
        f"Parsed {len(entries):_d} abstracts in {m}m {s}s.\n"
    )

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for i, abstract in enumerate(entries):
            with open(
                f"{save_dir}abstract_{i+1}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(abstract)

    return entries


### SPACY PIPELINE COMPONENTS ###
def normalize_bigram_span(
    bigram: Span, concat: Literal["norm_"] | Literal["lemma_"] = "norm_",
    lower: bool = True
) -> str:
    """Concatenate the normalized or lemmatized text of a bigram span.
    Optionally convert the result to lowercase.

    ---
    #### Note
    Replaces accents that are used as apostrophes with a standard apostrophe.
    (E.g. "student`s" -> "student's")
    """
    assert concat in ["norm_", "lemma_"]
    if concat == "lemma_":
        normalized = " ".join([tok.lemma_ for tok in bigram])
    else:
        normalized = " ".join([tok.norm_ for tok in bigram])
    normalized = patterns["accent_contraction_pat"].sub("'", normalized)
    return normalized.lower() if lower else normalized


def doc_candidate_counts(doc: Doc) -> dict[str, int]:
    """Custom `spacy.tokens.Doc`-extension method to identify & normalize
    counts of candidate bigrams in a document.

    Accessible as `spacy.tokens.Doc`-attribute: `doc._.candidate_counts`.

    ---
    ### Returns
    `dict[str, int]`: A dictionary, mapping normalized candidate bigrams to
    their respective occurrence counts in the `doc`.
    """
    # checks individual tokens in bigrams
    def is_valid_token(tok: Token) -> bool:
        def valid_attributes(tok: Token) -> bool:
            return not (tok.is_sent_end or tok.is_punct
                        or tok.is_quote or tok.is_digit)

        def invalid_punct(tok: Token) -> bool:
            return (patterns["invalid_punct"].search(tok.norm_) is not None
                    and not (tok.is_digit or tok.like_num)
                    and len(tok.norm_) > 1)

        def contains_artifact(tok: Token) -> bool:
            return (patterns["latex"].search(tok.norm_) is not None
                    or patterns["xml_html"].search(tok.norm_) is not None)

        return (valid_attributes(tok)
                and not contains_artifact(tok)
                and not invalid_punct(tok))

    # checks individual bigrams for candidate-hood
    def is_candidate_bigram(bigram: Span) -> bool:
        def valid_candidate_tokens(bigram: Span) -> bool:
            return all(is_valid_token(tok) for tok in bigram)

        def valid_pos_tags(bigram: Span) -> bool:
            return (bigram[0].pos_ in ["ADJ", "NOUN", "PROPN"]
                    and bigram[1].pos_ in ["NOUN", "PROPN"])

        def not_et_al(bigram: Span) -> bool:
            return (
                patterns["et_al"]["tok1"].search(bigram[0].norm_) is None
                and patterns["et_al"]["tok2"].search(bigram[1].norm_) is None
            )

        return (
            valid_candidate_tokens(bigram)
            and valid_pos_tags(bigram)
            and not_et_al(bigram)
        )

    # wraps the above functions to yield valid candidate bigrams
    def candidate_bigrams(doc: Doc) -> Iterator[Span]:
        """Checks individual `Token`s in all bigram-`Span`s in a `doc` for
        validity, yielding normalized, valid candidate bigrams."""
        for i in range(len(doc) - 1):
            if is_candidate_bigram(bigram := doc[i:i+2]):
                yield normalize_bigram_span(bigram)

    return dict(Counter(candidate_bigrams(doc)))


def merge_hyphenated_multiwords(doc: Doc) -> Doc:
    """Custom spacy-pipeline component to merge hyphenated multiword
    tokens.

    (E.g. `["fine", "-", "tuned"]` -> `["fine-tuned"]`)"""
    # find tokens in raw text, merge them into tokens and inherit POS tags
    with doc.retokenize() as retokenizer:
        for m in patterns["hyphenated_mw_pat"].finditer(
            doc.text
        ):
            if (span := doc.char_span(*m.span())) is not None:
                retokenizer.merge(doc[span.start:span.end])
    return doc


### VECTORIZATION ###
def vectorize_corpus(
    corpus: Iterator[Doc]
) -> Tuple[sparse.csr_matrix, dict[str, int]]:
    """Vectorize a corpus of `Doc`-objects and return it as a
    `scipy.sparse.csr_matrix` along with a corresponding vocab-dict-mapping.
    """
    vectorizer = DictVectorizer(dtype=np.int32)
    doc_term_matrix = vectorizer.fit_transform(
        doc_candidate_counts(doc) for doc in corpus
    )

    return doc_term_matrix, vectorizer.vocabulary_


### EVALUATION ###
def gold_terms(file_path: str = None) -> set[str]:
    """Load gold terminology from `gold_terminology_abstracts.txt`, stripping
    leading and trailing whitespaces from lines.

    ---
    ### Parameters
    `file_path: str`: Path to the file containing gold terminology terms.
    If `None`, the default path is used:
    `../data/gold_terminology_abstracts.txt`.

    ---
    ### Returns
    `set[str]`: A set of gold terminology terms, stripped of leading and
    trailing whitespace.
    """
    if file_path is None:
        file_path = os.path.abspath(os.path.join(
            "..", os.path.dirname(__file__), "data",
            "gold_terminology_abstracts.txt"
        ))
    else:
        file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File '{file_path}' does not exist."
        )
    elif not os.path.isfile(file_path):
        raise TypeError(
            f"Expected a file path but got '{file_path}'."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        gold = set(line.strip() for line in f.readlines())
    if len(gold) == 0:
        logger.error(
            f"No gold terminology terms found in '{file_path}'."
        )
    logger.info(
        f"Using {len(gold)} gold terms for evaluation."
    )
    return gold


def compute_prec_recall_f1(
    extracted_terms: set[str], gold: set[str] = None
) -> dict[str, float]:
    """Compute precision and recall scores for extracted terms against
    gold-terms, taken from `gold_terminology_abstracts.txt`.

    ---
    ### Parameters
    `extracted_terms: set[str]`: A set of extracted terms.

    ### Returns
    `dict[str, float]`: Dict with keys `"precision"`, `"recall"` & `"f1"`, each
    rounded to 6 decimal places.
    """
    def true_positives(extracted: set[str], gold: set[str]) -> int:
        return len(extracted.intersection(gold))

    def false_positives(extracted: set[str], gold: set[str]) -> int:
        return len(extracted.difference(gold))

    def false_negatives(extracted: set[str], gold: set[str]) -> int:
        return len(gold.difference(extracted))

    if not isinstance(extracted_terms, set):
        raise TypeError(
            "Expected 'extracted_terms' to be of type 'set' but got " +
            f"'{type(extracted_terms)}'."
        )
    elif not all(isinstance(term, str) for term in extracted_terms):
        raise TypeError(
            "Expected all elements in 'extracted_terms' to be of type 'str'."
        )

    gold = gold_terms() if gold is None else gold
    tp: int = true_positives(extracted_terms, gold)
    fp: int = false_positives(extracted_terms, gold)
    fn: int = false_negatives(extracted_terms, gold)
    prec: float = round(
        float(tp / (tp + fp)), 6
    ) if tp + fp > 0 else 0.0
    rec: float = round(
        float(tp / (tp + fn)), 6
    ) if tp + fn > 0 else 0.0
    f1: float = round(
        float(2 * tp / (2 * tp + fp + fn)), 6
    ) if tp + fp + fn > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def evaluate_run(
    extracted_terms: set[str], alpha: float, theta: float,
    gold_terms: set[str] = None
) -> dict[str, float | int]:
    """Compute summary metrics of a run.

    ---
    ### Parameters
    `extracted_terms: set[str]`: A set of extracted terms.

    `alpha: float`: The hyperparameter `alpha` used for weighting candidates'
    scores.

    `theta: float`: The hyperparameter `theta` used for term extraction.

    `gold_terms: set[str]`: A set of gold terminology terms. If `None`, terms
    are loaded from `gold_terminology_abstracts.txt`. (Default: `None`)

    ---
    ### Returns
    `metrics: dict` with keys:
    - `"alpha": float` The hyperparameter `alpha`
    - `"theta": float` The hyperparameter `theta`
    - `"precision": float` Precision of the extracted terms
    - `"recall": float` Recall of the extracted terms
    - `"f1": float` (Harmonic) F1-score of the extracted terms
    - `"n_terms": int` Number of extracted terms
    """
    metrics: dict[str, float] = compute_prec_recall_f1(
        extracted_terms, gold_terms
    )
    return {
        "alpha": alpha,
        "theta": theta,
        **metrics,
        "n_terms": len(extracted_terms)
    }


def save_terms(
    alpha: float, theta: float, extracted_terms: pd.Series,
    save_dir: str = "../output/", verbose: bool = True
) -> None:
    """Write extracted terms from a single run to a tsv-file in `save_dir`.

    ---
    ### Parameters
    `alpha: float`: The `alpha`-parameter used for extraction.

    `theta: float`: The `theta`-parameter used for extraction.

    `extracted_terms: pd.Series`: Series of extracted terms with their computed
    scores

    `save_dir: str`: Path to a directory in which to create a results-file.

    `verbose: bool`: Whether to log the save path.

    ---
    ### File format
    **File name**: `exp_{n}.tsv`

        (File names are incrementally numbered to avoid overwriting
        previous results.)
    ```
    alpha={alpha: float}
    theta={theta: float}
    terms[i]\tscores[i]
    terms[i+1]\tscores[i+1]
    ...
    ```

    ---
    ### Note
    Since computed scores are stored as `np.float64`, results are rounded to a
    precision of 20 decimal places.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif not os.path.isdir(save_dir):
        logger.error(
            f"Please provie a valid directory path. '{save_dir}' is not a " +
            "directory."
        )
    n: int = len(glob(f"{save_dir}exp_*.tsv"))
    save_path: str = f"{save_dir}exp_{n+1}.tsv"  # 1-indexed
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(
            f"alpha={alpha}\n" +
            f"theta={theta}\n"
        )
        for term, score in extracted_terms.items():
            f.write(f"{term}\t{score}\n")
    if verbose:
        logger.info(f"Saved extracted terms to '{save_path}'.")


def get_memory_usage():
    """Return the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB
