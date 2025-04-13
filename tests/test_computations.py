#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Fri, 28.02.25                                                     #
# ========================================================================== #

"""Tests for the `computations` module."""

import os
import sys
import logging
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.abspath(  # import src package
    os.path.join(os.path.dirname(__file__), '..')
))
from src.computations import (
    collection_frequencies, collection_probabilities,
    compute_domain_relevance,
    compute_domain_consensus,
    candidate_entropy, candidate_probability_distributions,
    compute_domain_consensus,
    compute_dr_dc,
    extract_terms
)
from src.utils import compute_prec_recall_f1, evaluate_run


def test_collection_frequencies(
    domain_matrix, ref_matrix, domain_cf, ref_cf
):
    """Test the computation of collection frequencies (i.e. total occurrences
    per candidate in each corpus).
    """
    assert np.all(domain_cf == collection_frequencies(domain_matrix))
    assert np.all(ref_cf == collection_frequencies(ref_matrix))


def test_relative_collection_frequencies(
    domain_cf, ref_cf, domain_rel_freqs, ref_rel_freqs
):
    """Test the computation of relative collection frequencies (i.e. the
    probability distributions of each candidate across the corpus).
    """
    assert np.all(
        np.isclose(collection_probabilities(domain_cf), domain_rel_freqs)
    )
    assert np.all(
        np.isclose(collection_probabilities(ref_cf), ref_rel_freqs)
    )


def test_domain_relevance(
    domain_matrix, ref_matrix, vocab, candidates_domain_relevance
):
    """Test the computation of domain relevance scores."""
    assert np.all(np.isclose(compute_domain_relevance(
        domain_matrix, ref_matrix, vocab
    ), candidates_domain_relevance.to_numpy()))


def test_candidate_probability_distributions(
    domain_matrix, candidate_distributions
):
    """Test that the generator yields the correct probability distributions."""
    # check that the generator yields correctly computed probability
    # distributions (without explicit zeros)
    for expected, computed in zip(
        candidate_distributions,
        candidate_probability_distributions(domain_matrix)
    ):
        assert expected.shape == computed.shape
        assert np.all(np.isclose(expected, computed))
        assert np.all(computed > 0.0)


def test_candidate_entropy(
    candidate_distributions, candidates_domain_consensus
):
    """Test the entropy computation of candidates."""
    # entropy is 0 if all probability mass is on one document
    lower_bound = 0.0
    # if a term occurs with the same probability in all documents, the
    # entropy reduces to the log2 of the number of documents: log2(4) = 2
    upper_bound: float = np.log2(4)

    for dist, expected in zip(
        candidate_distributions, candidates_domain_consensus
    ):
        computed: float = candidate_entropy(dist)
        assert computed == pytest.approx(expected)
        assert lower_bound <= computed <= upper_bound


def test_domain_consensus(domain_matrix, vocab, candidates_domain_consensus):
    """Test the computation of domain consensus scores."""
    assert np.all(np.isclose(compute_domain_consensus(
        domain_matrix, vocab, parallel=False
    ).to_numpy(), candidates_domain_consensus))


def test_dr_dc_wrapper(
    domain_matrix, ref_matrix, vocab,
    candidates_domain_relevance, candidates_domain_consensus
):
    """Test if the wrapper function works."""
    df = pd.DataFrame(
        candidates_domain_relevance, columns=["DR"],
        index=candidates_domain_relevance.index
    ).assign(DC=candidates_domain_consensus)
    computed_df: pd.DataFrame = compute_dr_dc(
        domain_matrix, ref_matrix, vocab
    )
    assert computed_df.columns.tolist() == ["DR", "DC"]
    assert computed_df.shape == (candidates_domain_relevance.shape[0], 2)
    for col in computed_df.columns:
        assert np.all(np.isclose(
            computed_df[col].to_numpy(), df[col].to_numpy()
        ))
    assert np.all(computed_df.index == df.index)


def test_scoring_and_extraction(
    candidates_domain_relevance, candidates_domain_consensus, candidate_scores,
    caplog
):
    """Test the computation of terminology-scores and extraction of
    terminology."""
    df = pd.DataFrame(
        candidates_domain_relevance, columns=["DR"],
        index=candidates_domain_relevance.index
    ).assign(DC=candidates_domain_consensus)

    # check that invalid alpha and theta values are caught
    invalid_alpha = [-1.0, 1.1, "foo", 5]
    valid_alpha = [0.65, 0, 1.0, 0.000001]
    invalid_theta = [0.0, np.inf, -0.1, "bar"]
    valid_theta = [0.1, 1.0, 1, 100.5]
    with caplog.at_level(logging.ERROR):  # alpha-validation
        for invalid, valid in zip(invalid_alpha, valid_theta):
            extract_terms(df, alpha=invalid, theta=valid)
            assert "Invalid" in caplog.text and "alpha" in caplog.text
    caplog.clear()  # clear log
    with caplog.at_level(logging.ERROR):  # theta-validation
        for valid, invalid in zip(valid_alpha, invalid_theta):
            extract_terms(df, alpha=valid, theta=invalid)
        assert len(caplog.records) == 3

    # check that function returns df with expected columns & shape
    computed_df = extract_terms(df, alpha=valid_alpha[0], theta=valid_theta[0])
    assert computed_df.shape == df.shape
    assert computed_df.columns.tolist() == ["DR", "DC", "score", "is_term"]

    # check that terminology-score is computed correctly
    not_terms: set[str] = {  # with alpha=0.65, theta=0.1
        "preliminary experiments", "human texts", "repetitive tasks",
        "pervasive presence", "first attempt", "ai counterpart",
        "measurable definition", "many category"
    }
    computed_df = extract_terms(df, alpha=0.65, theta=0.1)
    assert np.all(np.isclose(computed_df.score.to_numpy(), candidate_scores))
    assert set(
        computed_df[~computed_df["is_term"]].index.tolist()
    ) == not_terms
    assert len(
        computed_df[computed_df["is_term"]]
    ) == len(df) - len(not_terms)


def test_evaluation(sample_terms, sample_gold_terms):
    """Test the evaluation of terminology extraction for correct computation
    of precision, recall & (harmonic) F1 values."""
    true_positives = sample_terms.intersection(sample_gold_terms)
    false_positives = sample_terms.difference(sample_gold_terms)
    false_negatives = sample_gold_terms.difference(sample_terms)
    expected_precision: float = len(true_positives) / (
        (len(true_positives) + len(false_positives))
    )
    expected_recall: float = len(true_positives) / (
        len(true_positives) + len(false_negatives)
    )
    expected_f1: float = 2 * (expected_precision * expected_recall) / (
        expected_precision + expected_recall
    )
    expected_eval: dict[str, float] = {
        "precision": round(expected_precision, 6),
        "recall": round(expected_recall, 6),
        "f1": round(expected_f1, 6)
    }
    computed_eval: dict[str, float] = compute_prec_recall_f1(
        extracted_terms=sample_terms, gold=sample_gold_terms
    )
    assert set(expected_eval.keys()) == set(computed_eval.keys())
    for key in expected_eval.keys():
        # check that computed values are correctly rounded
        assert computed_eval[key] == expected_eval[key]

    # provide values to match function-signature
    n_terms = len(sample_terms)
    alpha = 0.65
    theta = 0.1

    expected_results = {
        **expected_eval, "n_terms": n_terms, "alpha": alpha, "theta": theta
    }
    computed_results: dict[str, float | int] = evaluate_run(
        extracted_terms=sample_terms, gold_terms=sample_gold_terms,
        alpha=alpha, theta=theta
    )
    assert set(expected_results.keys()) == set(computed_results.keys())
    for key in expected_results.keys():
        assert computed_results[key] == expected_results[key]
