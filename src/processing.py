#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Sat, 01.03.25                                                     #
# ========================================================================== #

"""Load & process raw text data, extract & normalize candidate bigrams and
vectorize domain & reference corpora."""

import logging
from multiprocessing import cpu_count
import numpy as np
import os
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import sys
from time import time
from typing import Iterator, Tuple
sys.path.insert(0, os.path.abspath(  # add src to path
    os.path.join(os.path.dirname(__file__), '.')
))
from utils import doc_candidate_counts, merge_hyphenated_multiwords  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%H:%M:%S"
)
logger: logging.Logger = logging.getLogger(__name__)


class SpacyProcessor:
    def __init__(self, model: str = "en_core_web_md"):
        self.spacy: Language = spacy.load(
            model,
            disable=["parser", "lemmatizer", "textcat", "ner"]
        )
        self.spacy.enable_pipe("senter")
        Language.component(  # register custom processing-component...
            "merge_hyphenated_multiwords", func=merge_hyphenated_multiwords,
            retokenizes=True
        )
        self.spacy.add_pipe(  # and add it to the pipeline
            "merge_hyphenated_multiwords", after="attribute_ruler"
        )
        self.spacy.add_pipe("doc_cleaner", last=True, config={
            "attrs": {"tensor": None}  # remove doc-tensor to save memory
        })
        if not Doc.has_extension("candidate_counts"):  # doc._.candidate_counts
            Doc.set_extension(
                "candidate_counts", getter=doc_candidate_counts
            )
        logger.info(
            "Initialized spaCy pipeline with model " +
            f"'\033[3m{self.spacy.meta['name']}\033[0m' and the following " +
            f"components:\n{' ' * 17}{', '.join(self.spacy.pipe_names)}\n"
        )

    def __call__(
        self, data: list[str] | str,
        batch_size: int = 1000, n_process: int = cpu_count() - 2
    ) -> Iterator[Doc]:
        if not isinstance(data, (str, list)):
            raise TypeError(
                "data must be a string or an iterable of strings."
            )
        elif isinstance(data, str):
            yield self.spacy(data)
        elif (n := len(data)) < batch_size:
            yield from self.spacy.pipe(data, batch_size=n)
        else:
            yield from self.spacy.pipe(
                data, batch_size=batch_size, n_process=n_process
            )


def process_and_vectorize_corpora(
    domain_corpus: list[str], ref_corpus: list[str],
    spacy_model: str = "en_core_web_md", batch_size: int = 1000,
    n_process: int = cpu_count() - 2
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, int]]:
    """Process domain & reference corpora and vectorize for subsequent
    computations.

    ---
    ### Parameters
    `domain_corpus: list[str]`: A list of raw text documents from the domain
    corpus.

    `ref_corpus: list[str]`: A list of raw text documents from the reference
    corpus.

    `spacy_modeL: str`: The spaCy model to use for processing the corpora.
    (default: "en_core_web_md")

    `batch_size: int`: The number of documents to process per batch.
    (default: 1000)

    `n_process: int`: The number of processes to use for parallel processing.
    (defaults to utilizing all but 2 CPU cores)

    ### Returns
    `Tuple` of:
    - `domain_doc_candidate_matrix: scipy.sparse.csr_matrix`: Absolute
    occurrence counts of normalized candidate bigrams in the domain corpus.

    - `ref_doc_candidate_matrix: scipy.sparse.csr_matrix`: Absolute occurrence
    counts of domain candidates in the reference corpus. (Counts of
    OOV-candidates that aren't present in the domain corpus are omitted and
    set to zero.)

    - `vocab: dict[str, int]`: A dictionary mapping normalized
    candidate-`string`s to their respective column indices in the matrices.
    (Indices are identical between both matrices.)

    ---
    ### Note on normalization of bigrams
    By default, the following normalization steps are applied to bigrams in
    both corpora:
    - filtering out bigrams with POS-tags other than `ADJ-NOUN`, `ADJ-PROPN`,
    `NOUN-NOUN`, `NOUN-PROPN`
    - filtering out bigrams for which any token is punctuation or a digit
    - filtering out bigrams that contain LaTeX, XML or HTML artifacts
    - filtering out "et al"-bigrams
    - lowercasing all tokens
    - hyphenated multiword tokens are treated as single tokens (e.g.
    `["fine", "-", "tuned"]` -> `["finetuned"]`)
    """
    nlp: SpacyProcessor = SpacyProcessor(model=spacy_model)
    vectorizer = DictVectorizer(sparse=True, dtype=np.int32)
    logger.info("Processing & vectorizing domain and reference corpora...")

    # fit vectorizer on candidates from domain corpus to learn vocabulary
    t0 = time()
    domain_doc_term_matrix = vectorizer.fit_transform(
        # process domain texts, identifying & normalizing candidates in docs
        doc_candidate_counts(doc) for doc in nlp(
            domain_corpus, batch_size=batch_size, n_process=n_process
        )
    ).sorted_indices()
    t1 = time()
    m, s = [int(t) for t in divmod(t1 - t0, 60)]
    logger.info(f"Vectorized domain corpus in {m}m {s}s.")

    # transform reference corpus using learned domain vocabulary, omitting
    # OOV-terms from reference doc-term-matrix
    ref_doc_term_matrix = vectorizer.transform(
        # process reference texts, identifying & normalizing candidates in docs
        doc_candidate_counts(doc) for doc in nlp(
            ref_corpus, batch_size=batch_size, n_process=n_process
        )
    ).sorted_indices()
    m, s = [int(t) for t in divmod(time() - t1, 60)]
    logger.info(f"Vectorized reference corpus in {m}m {s}s.")

    # log processing summary
    n_total: int = domain_doc_term_matrix.shape[1]
    n_shared: int = len(
        np.asarray(ref_doc_term_matrix.sum(axis=0)).nonzero()[-1]
    )
    perc_shared: float = round(n_shared / n_total * 100, 2)
    m_total, s_total = [int(t) for t in divmod(time() - t0, 60)]
    logger.info(
        "### \033[1mPROCESSING DONE\033[0m - Total processing time: " +
        f"\033[1m{m_total}m {s_total}s\033[0m. ###\n{' ' * 17}Identified " +
        f"{n_total:_d}\t\tunique candidates in domain corpus,\n{' ' * 17}" +
        f"of which     {n_shared:_d} ({perc_shared}%)\tare also present in " +
        "the reference corpus.\n"
    )

    return (
        domain_doc_term_matrix, ref_doc_term_matrix, vectorizer.vocabulary_
    )


### EXAMPLE USAGE ###
if __name__ == "__main__":
    nlp = SpacyProcessor()
    domain_sample: list[str] = [
        "In this paper, we introduce the concept of Semantic Masking, where semantically coherent surrounding text (the haystack) interferes with the retrieval and comprehension of specific information (the needle) embedded within it. We propose the Needle-in-a-Haystack-QA Test, an evaluation pipeline that assesses LLMs' long-text capabilities through question answering, explicitly accounting for the Semantic Masking effect. We conduct experiments to demonstrate that Semantic Masking significantly impacts LLM performance more than text length does. By accounting for Semantic Masking, we provide a more accurate assessment of LLMs' true proficiency in utilizing extended contexts, paving the way for future research to develop models that are not only capable of handling longer inputs but are also adept at navigating complex semantic landscapes.",  # noqa: E501
        "Nowadays, AI is present in all our activities. This pervasive presence is perceived as a threat by many category of users that might be substituted by their AI counterpart. While the potential of AI in handling repetitive tasks is clear, the potentials of its creativeness is still misunderstood. We believe that understanding this aspects of AI can transform a threat into an opportunity. This paper is a first attempt to provide a measurable definition of creativeness. We applied our definition to AI and human generated texts, proving the viability of the proposed approach. Our preliminary experiments show that human texts are more creative.",  # noqa: E501
        "While MSA and some dialects of Arabic have been extensively studied in NLP, Middle Arabic is still very much unknown to the field. However, Middle Arabic holds issues that are still not covered: it is characterized by variation since it mixes standard features, colloquial ones, as well as features that belong to neither of the two. Here, we introduce a methodology to identify, extract and rank variations of 13 manually retrieved formulas. Those formulas come from the nine first booklets of S ̄IRAT AL-MALIK AL-Z. ̄AHIR BAYBAR S., a corpus of Damascene popular literature written in Middle Arabic and composed of 53,843 sentences. In total, we ranked 20, sequences according to their similarity with the original formulas on multiple linguistic layers. We noticed that the variations in these formulas occur in a lexical, morphological and graphical level, but in opposition, the semantic and syntactic levels remain strictly invariable.",  # noqa: E501
        "In this paper, we introduce the concept of Semantic Masking, where semantically coherent surrounding text (the haystack) interferes with the retrieval and comprehension of specific information (the needle) embedded within it. We propose the Needle-in-a-Haystack-QA Test, an evaluation pipeline that assesses LLMs' long-text capabilities through question answering, explicitly accounting for the Semantic Masking effect. We conduct experiments to demonstrate that Semantic Masking significantly impacts LLM performance more than text length does. By accounting for Semantic Masking, we provide a more accurate assessment of LLMs' true proficiency in utilizing extended contexts, paving the way for future research to develop models that are not only capable of handling longer inputs but are also adept at navigating complex semantic landscapes.",  # noqa: E501
    ]
    ref_sample: list[str] = [
        "CHINA DAILY SAYS VERMIN EAT 7-12 PCT GRAIN STOCKS\n  A survey of 19 provinces and seven cities\n  showed vermin consume between seven and 12 pct of China's grain\n  stocks, the China Daily said.\n      It also said that each year 1.575 mln tonnes, or 25 pct, of\n  China's fruit output are left to rot, and 2.1 mln tonnes, or up\n  to 30 pct, of its vegetables. The paper blamed the waste on\n  inadequate storage and bad preservation methods.\n      It said the government had launched a national programme to\n  reduce waste, calling for improved technology in storage and\n  preservation, and greater production of additives. The paper\n  gave no further details.\n  \n\n",  # noqa: E501
        "THAI TRADE DEFICIT WIDENS IN FIRST QUARTER\n  Thailand's trade deficit widened to 4.5\n  billion baht in the first quarter of 1987 from 2.1 billion a\n  year ago, the Business Economics Department said.\n      It said Janunary/March imports rose to 65.1 billion baht\n  from 58.7 billion. Thailand's improved business climate this\n  year resulted in a 27 pct increase in imports of raw materials\n  and semi-finished products.\n      The country's oil import bill, however, fell 23 pct in the\n  first quarter due to lower oil prices.\n      The department said first quarter exports expanded to 60.6\n  billion baht from 56.6 billion.\n      Export growth was smaller than expected due to lower\n  earnings from many key commodities including rice whose\n  earnings declined 18 pct, maize 66 pct, sugar 45 pct, tin 26\n  pct and canned pineapples seven pct.\n      Products registering high export growth were jewellery up\n  64 pct, clothing 57 pct and rubber 35 pct.\n  \n\n",  # noqa: E501
        "Nowadays, AI is present in all our activities. This pervasive presence is perceived as a threat by many category of users that might be substituted by their AI counterpart. While the potential of AI in handling repetitive tasks is clear, the potentials of its creativeness is still misunderstood. We believe that understanding this aspects of AI can transform a threat into an opportunity. This paper is a first attempt to provide a measurable definition of creativeness. We applied our definition to AI and human generated texts, proving the viability of the proposed approach. Our preliminary experiments show that human texts are more creative."  # noqa: E501
    ]
    domain_matrix, ref_matrix, vocab = process_and_vectorize_corpora(
        domain_corpus=domain_sample, ref_corpus=ref_sample, n_process=1
    )
