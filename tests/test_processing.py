# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Tue, 04.03.25                                                     #
# ========================================================================== #

"""Tests for the `processing` module."""

import numpy as np
import os
import pytest
from spacy.tokens import Doc
import sys
sys.path.insert(0, os.path.abspath(  # import src package
    os.path.join(os.path.dirname(__file__), '..')
))
from processing import process_and_vectorize_corpora
from utils import (
    parse_bibtex_abstracts, merge_hyphenated_multiwords, doc_candidate_counts
)


def test_bibtex_parsing(raw_bibtex, parsed_bibtex_en, parsed_bibtex_all):
    """Test that BibTeX-entries are correctly parsed."""
    assert (parsed_en := parse_bibtex_abstracts(
        raw_bibtex, english_only=True
    )) == parsed_bibtex_en
    assert (parsed_all := parse_bibtex_abstracts(
        raw_bibtex, english_only=False
    )) == parsed_bibtex_all
    assert len(parsed_en) == 2
    assert len(parsed_all) == 3


def test_merge_hyphenated_multiwords(
    hyphenated_multiword_text, spacy_processor
):
    """Test that the custom pipeline-component correctly merges hyphenated
    multiwords."""
    nlp = spacy_processor
    multi_words = ["needle-in-a-haystack-qa", "long-text"]

    # process w/o custom component
    nlp.spacy.disable_pipe("merge_hyphenated_multiwords")
    doc = nlp.spacy(hyphenated_multiword_text)
    assert not any([
        token.lower_ in multi_words
        for token in doc
    ])

    # now manually call component
    mw_doc_tokens = [
        token.lower_ for token in merge_hyphenated_multiwords(doc)
    ]
    assert all([mw_tok in mw_doc_tokens for mw_tok in multi_words])
    nlp.spacy.enable_pipe("merge_hyphenated_multiwords")  # re-enable


def test_doc_candidate_counts(sample_doc, sample_doc_counts):
    """Test that candidate bigrams in a document are correctly identified,
    normalized and counted."""
    # check that doc-extension is set
    assert not sample_doc.has_extension("candidate_counts")

    # check that required token-annotations are present
    sample_tok = sample_doc[0]
    assert sample_tok.i == 0
    assert sample_tok.norm_ != ""
    assert sample_tok.is_sent_end in [True, False]
    assert sample_tok.is_punct in [True, False]
    assert sample_tok.is_quote in [True, False]
    assert sample_tok.is_digit in [True, False]

    # check that computed candidate-counts are correct
    computed_counts = doc_candidate_counts(sample_doc)
    assert set(computed_counts.items()) == set(sample_doc_counts.items())
    Doc.set_extension("candidate_counts", getter=doc_candidate_counts)


def test_spacy_pipeline(spacy_processor):
    """Test that the spaCy-pipeline is applying the expected processing
    steps."""
    expected_components: list[str] = [
        "tok2vec", "tagger", "senter", "attribute_ruler",
        "merge_hyphenated_multiwords", "doc_cleaner"
    ]
    assert spacy_processor.spacy.pipe_names == expected_components


def test_vectorization(
    domain_sample, domain_matrix, ref_sample, ref_matrix, vocab
):
    """Test that the `process_and_vectorize_corpora`-wrapper correctly
    vectorizes the corpora."""
    computed_dom_matrix, computed_ref_matrix, computed_vocab = process_and_vectorize_corpora(  # noqa: E501
        domain_sample, ref_sample, batch_size=4, n_process=1
    )
    computed_dom_matrix = np.asarray(computed_dom_matrix.todense())
    computed_ref_matrix = np.asarray(computed_ref_matrix.todense())
    assert np.all(computed_dom_matrix == domain_matrix)
    assert np.all(computed_ref_matrix == ref_matrix)
    assert computed_vocab == vocab
    assert computed_dom_matrix.shape[1] == len(computed_vocab) == len(vocab)
    assert np.all(computed_dom_matrix[0] == computed_dom_matrix[-1])
    assert np.all(computed_dom_matrix[1] == computed_ref_matrix[-1])
