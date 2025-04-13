#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Tue, 25.02.25                                                     #
# ========================================================================== #

from __future__ import annotations
from datetime import datetime
from joblib import Parallel, delayed
import logging
import numpy as np
import os
import pandas as pd
from scipy import sparse
from scipy.stats import entropy
from shutil import get_terminal_size
import sys
from time import time
from tqdm import tqdm
from typing import Generator, Tuple

sys.path.insert(0, os.path.abspath(  # add src to path
    os.path.join(os.path.dirname(__file__), '.')
))
OUT_DIR = os.path.abspath(  # only for logging
    os.path.join("..", os.path.dirname(__file__), "output")
)

from utils import (  # noqa: E402
    evaluate_run, get_memory_usage, gold_terms, save_terms
)


logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S"
)


def collection_frequencies(matrix: sparse.csr_matrix) -> np.ndarray:
    """Compute the absolute collection frequencies of candidates in a corpus.

    ---
    ### Parameters
    `matrix: sparse.csr_matrix`: The vectorized corpus with shape:
    `(n_docs, n_candidates)`

    ### Returns
    `np.ndarray[np.int16]`: An array of absolute collection frequencies with
    shape: `(n_candidates,)`
    """
    return np.asarray(matrix.sum(axis=0)).flatten()


def collection_probabilities(collection_frequencies: np.ndarray) -> np.ndarray:
    """Compute the conditional probabilities of candidates in a corpus.

    The probability of a candidate is approximated by the ratio of its
    collection frequency to the total number of candidates in the corpus. (I.e.
    its relative frequency: `cf / total`)

    ---
    ### Parameters
    `collection_frequencies: np.ndarray[np.int16]`: An array of (absolute)
    collection frequencies with shape: `(n_candidates,)`

    ### Returns
    `np.ndarray[np.float64]`: An array of candidates' conditional probabilities
    with shape: `(n_candidates,)`
    """
    return collection_frequencies / collection_frequencies.sum()


def compute_domain_relevance(
    domain_matrix: sparse.csr_matrix, ref_matrix: sparse.csr_matrix,
    vocab: dict[str, int]
) -> pd.Series:
    """Compute the domain relevance of all candidates in the domain corpus.

    Domain relevance is defined as the ratio of a candidate's conditional
    probability in the domain corpus to the sum of its conditional
    probabilities across both the domain & reference corpora:

    `DR[t] = P(t | D_domain) / (P(t | D_domain) + P(t | D_ref))`

    ---
    ### Parameters
    `domain_matrix: sparse.csr_matrix`: The vectorized domain corpus with
    shape: `(n_domain_docs, n_candidates)`

    `ref_matrix: sparse.csr_matrix`: The vectorized reference corpus with
    shape: `(n_ref_docs, n_candidates)`

    `vocab: dict[str, int]`: A dictionary, mapping candidates to their
    corresponding column-indices in the domain & reference matrices.

    ### Returns
    `pd.Series`: A Series of domain relevance scores (`np.float64`), indexed by
    candidate-bigrams' normalized string representations.
    """
    if not all(isinstance(m, sparse.csr_matrix)
               for m in [domain_matrix, ref_matrix]):
        logger.error(
            "Invalid type for matrices. Must be of type sparse.csr_matrix."
        )
        return
    if not (all(isinstance(k, str) for k in vocab.keys())
            and all(isinstance(v, int) for v in vocab.values())):
        logger.error("Invalid type for vocab. Must be a dict[str, int].")
        return
    elif len(vocab) != domain_matrix.shape[1]:
        logger.error(
            "Mismatch in number of candidates and vocabulary size. Found " +
            f"{domain_matrix.shape[1]} candidates but {len(vocab)} " +
            "vocabulary terms."
        )
        return

    t0 = time()
    logger.debug("Computing domain relevance...")

    # compute candidates' collection frequencies
    domain_cf = collection_frequencies(domain_matrix)
    ref_cf = collection_frequencies(ref_matrix)

    # compute candidates' conditional probabilities (i.e. relative frequencies)
    domain_prob: np.ndarray = collection_probabilities(domain_cf)
    ref_prob: np.ndarray = collection_probabilities(ref_cf)

    # compute domain relevance from conditional probabilities
    dr = pd.Series(domain_prob / (domain_prob + ref_prob),
                   dtype=np.float64, name="DR", index=sorted(
                       vocab.keys(), key=lambda candidate: vocab[candidate]
                    ))

    m, s = [int(t) for t in divmod(time() - t0, 60)]
    logger.info(f"Computed domain relevance in {m}m {s}s.")
    return dr


def candidate_entropy(candidate_prob_dist, base=2) -> float:
    """Compute the entropy of a candidate's probability distribution across
    domain documents.

    ---
    ### Parameters
    `candidate_prob_dist: np.ndarray[float]`: The probability distribution of a
    candidate across all documents in the corpus.

    ### Returns
    `float`: The (Shannon) entropy of the candidate's probability distribution
    expressed in bits.

    ---
    ### Note
    Entropy is computed with `scipy.stats.entropy`_, using the formula:

    >    `H(P(t, d[j])) = -Σ P(t, d[j]) * log2(P(t, d[j]))`,

    which is mathematically equivalent to the formula from `Velardi et al.
    (2001)`_:

    >    `H(P(t, d[j])) = Σ P(t, d[j]) * log2(1 / P(t, d[j]))`.

    .. _scipy.stats.entropy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html#scipy.stats.entropy
    .. _Velardi et al. (2001): https://aclanthology.org/W01-1005/
    """
    return entropy(candidate_prob_dist, base=base)


def candidate_probability_distributions(
    domain_matrix: sparse.csc_matrix
) -> Generator[np.ndarray, None, None]:
    """Generator to yield candidates' (sparse) probability distributions across
    domain documents.

    ---
    ### Parameters
    `domain_matrix: sparse.csc_matrix`: The vectorized domain corpus with
    shape: `(n_domain_docs, n_candidates)`

    ### Yields
    `candidate_prob_dist: np.ndarray[float]`: The probability distribution of a
    candidate across all documents in the corpus. (non-zero values only)

    ---
    ### Note
    To minimize memory usage, only non-zero values in candidates' probability
    distributions are considered for entropy computations by leveraging the
    sparse format of the vectorized matrix. This avoids NaN values due to
    multiplication/ division by zero (`xlogx(0)`) while producing
    mathematically equivalent results. Yielding the distributions rather than
    keeping them in memory further reduces memory consumption.
    """
    if not isinstance(domain_matrix, sparse.csc_matrix):
        logger.error(
            "Invalid type for domain_matrix. Must be a sparse matrix in CSC-" +
            "format."
        )
        return

    # normalize probabilities by column-sums
    col_sums: np.ndarray = np.asarray(domain_matrix.sum(axis=0)).ravel()
    probs: np.ndarray = domain_matrix.data / col_sums[
        domain_matrix.nonzero()[1]
    ]

    # yield probability distributions per candidate (i.e. per column)
    for j in range(domain_matrix.shape[1]):
        start, end = domain_matrix.indptr[j], domain_matrix.indptr[j+1]
        if (
            start < len(domain_matrix.data)
            and end <= len(domain_matrix.data)
            and start != end
        ):
            yield probs[start:end]  # yield non-zero probabilities


def compute_domain_consensus(
    domain_matrix: sparse.csr_matrix | sparse.csc_matrix,
    vocab: dict[str, int], parallel: bool = True, n_jobs: int = -1
) -> pd.Series:
    """Compute the domain consensus of candidates in the domain corpus.

    ---
    ### Parameters
    `domain_matrix: sparse.csr_matrix | sparse.csc_matrix`: The vectorized
    domain corpus with shape: `(n_domain_docs, n_candidates)`

    `vocab: dict[str, int]`: A dictionary, mapping candidates to their
    corresponding column-indices in the domain matrix.

    `parallel: bool`: Whether to compute entropy in parallel. (default: `True`)

    `n_jobs: int`: The number of CPU cores to use for parallel computation.

    ### Returns
    `pd.Series`: A Series of domain consensus scores (`np.float64`), indexed by
    candidate-bigrams' normalized string representations.
    """
    if not isinstance(domain_matrix, (sparse.csr_matrix, sparse.csc_matrix)):
        logger.error(
            "Invalid type for domain_matrix. Must be a sparse matrix."
        )
        return
    if not (all(isinstance(k, str) for k in vocab.keys())
            and all(isinstance(v, int) for v in vocab.values())):
        logger.error("Invalid type for vocab. Must be a dict[str, int].")
        return
    elif len(vocab) != domain_matrix.shape[1]:
        logger.error(
            "Mismatch in number of candidates and vocabulary size. Found " +
            f"{domain_matrix.shape[1]} candidates but {len(vocab)} " +
            "vocabulary terms."
        )
        return

    logger.debug(f"Initial memory usage: {get_memory_usage():.2f} MB")
    start_time = time()
    domain_matrix = domain_matrix.tocsc()

    if not parallel:
        col_sums = np.asarray(domain_matrix.sum(axis=0)).ravel()

        # Compute probabilities for each term across domain documents
        probs: np.ndarray = domain_matrix.data / col_sums[
            domain_matrix.nonzero()[1]
        ]

        # Reshape probabilities into a 2D array (columns x documents)
        prob_matrix = sparse.csc_matrix(
            (probs, domain_matrix.indices, domain_matrix.indptr),
            shape=domain_matrix.shape
        )
        entropies: np.ndarray = np.zeros(domain_matrix.shape[1])
        for j in tqdm(
            range(domain_matrix.shape[1]), desc="Computing domain consensus",
            unit="candidates", colour="cyan"
        ):
            start, end = domain_matrix.indptr[j], domain_matrix.indptr[j+1]
            if (
                start < len(domain_matrix.data)
                and end <= len(domain_matrix.data)
                and start != end
            ):
                entropies[j] = entropy(prob_matrix.data[start:end], base=2)
    else:
        entropies = Parallel(n_jobs=n_jobs)(
            delayed(candidate_entropy)(term_prob_slice) for term_prob_slice in
            tqdm(
                candidate_probability_distributions(domain_matrix),
                desc="Computing domain consensus", unit="candidates",
                colour="cyan", total=domain_matrix.shape[1]
            )
        )
        entropies = np.array(entropies, dtype=np.float64)

    m, s = [int(t) for t in divmod(time() - start_time, 60)]
    logger.debug(
        f"Computed domain consensus in {m}m {s}s." +
        f"Final memory usage: {get_memory_usage():.2f} MB\n" +
        f"Entropy calculation complete. Shape of result: {entropies.shape}\n" +
        f"Min entropy: {entropies.min()}\n" +
        f"Min entropy (without -inf): {entropies[entropies > -np.inf].min()}" +
        f"\nMax entropy (without -inf): {entropies[entropies > -np.inf].max()}"
        + f"\nUpper entropy bound: {np.log2(domain_matrix.shape[0])}"
    )
    return pd.Series(entropies, name="DC", index=sorted(
        vocab.keys(), key=lambda candidate: vocab[candidate]
    ), dtype=np.float64)


def compute_dr_dc(
    domain_matrix: sparse.csr_matrix | sparse.csc_matrix,
    ref_matrix: sparse.csr_matrix | sparse.csc_matrix, vocab: dict[str, int],
    parallel: bool = True, n_jobs: int = -1, save_dir: str = None
) -> pd.DataFrame:
    """Convenience wrapper to compute domain relevance & domain consensus,
    given the vectorized domain & reference corpora.

    ---
    ### Parameters
    `domain_matrix: sparse.csr_matrix | sparse.csc_matrix`: The vectorized
    domain corpus.

    `ref_matrix: sparse.csr_matrix | sparse.csc_matrix`: The vectorized
    reference corpus.

    `vocab: dict[str, int]`: A dictionary, mapping candidates to their
    corresponding column-indices in the domain & reference matrices.

    `parallel: bool`: Whether to compute domain consensus in parallel.
    (default: `True`)

    `n_jobs: int`: The number of CPU cores to use for parallel computation
    of domain consensus. (default: `-1`)

    `save_dir: str`: The directory to save the computed DR & DC scores to.
    (default: None)

    ---
    ### Returns
    `pd.DataFrame`: A DataFrame containing candidates with their respective
    domain relevance (`"DR"`) & domain consensus (`"DC"`) scores.
    """
    if not isinstance(domain_matrix, (sparse.csr_matrix, sparse.csc_matrix)):
        logger.error(
            "Invalid type for domain_matrix. Must be a sparse matrix."
        )
        return
    if not isinstance(ref_matrix, (sparse.csr_matrix, sparse.csc_matrix)):
        logger.error(
            "Invalid type for ref_matrix. Must be a sparse matrix."
        )
        return
    if domain_matrix.shape[1] != ref_matrix.shape[1]:
        logger.error(
            "Mismatch in number of candidates between domain & reference " +
            f"matrices. Found {domain_matrix.shape[1]} candidates in domain " +
            f"matrix but {ref_matrix.shape[1]} in reference matrix."
        )
        return
    elif len(vocab) != domain_matrix.shape[1]:
        logger.error(
            "Mismatch in number of candidates and vocabulary size. Found " +
            f"{domain_matrix.shape[1]} candidates but {len(vocab)} " +
            "vocabulary terms."
        )
        return
    if not (all(isinstance(k, str) for k in vocab.keys())
            and all(isinstance(v, int) for v in vocab.values())):
        logger.error("Invalid type for vocab. Must be a dict[str, int].")
        return

    t0 = time()
    domain_relevance: pd.Series = compute_domain_relevance(
        domain_matrix.tocsr().sorted_indices() if isinstance(
            domain_matrix, sparse.csc_matrix
        ) else domain_matrix, ref_matrix, vocab
    )
    domain_consensus: pd.Series = compute_domain_consensus(
        domain_matrix, vocab, parallel=parallel, n_jobs=n_jobs
    )
    df = pd.DataFrame(
        domain_relevance, columns=["DR"], index=domain_relevance.index
    ).assign(
        DC=domain_consensus
    )
    m, s = [int(t) for t in divmod(time() - t0, 60)]
    logger.info(
        "### \033[1mDR & DC-COMPUTATIONS DONE\033[0m - Total computation " +
        f"time: \033[1m{m}m {s}s\033[0m. ###\n"
    )
    if not sys.stdout.isatty():  # plot histogram unless running in terminal
        try:
            df.hist(
                column=["DR", "DC"],
                **{"histtype": "stepfilled", "log": True, "stacked": True},
                sharey=True, bins=15
            )
        except Exception as e:
            logger.warning(
                f"Failed to plot histogram of DR & DC.\nError: {e}"
            )
            pass
    if save_dir:
        try:
            if not os.path.exists(save_dir := os.path.abspath(save_dir)):
                os.makedirs(save_dir)
            tm_stamp: str = datetime.now().strftime("%d-%m-%y_%H-%M")
            df.to_csv(f"{save_dir}/dr_dc_{tm_stamp}.csv", encoding="utf-8")
            logger.info(
                f"Saved DR & DC scores to '{save_dir}/dr_dc_{tm_stamp}.csv'."
            )
        except Exception as e:
            logger.error(
                f"Failed to save DR & DC scores to '{save_dir}'.\nError: {e}"
            )
            pass
    return df


def extract_terms(
    candidate_metrics: pd.DataFrame, alpha: float, theta: float
) -> pd.DataFrame:
    """Extract domain terminology based on candidates' pre-computed domain
    relevance & domain consensus, using hyperparameters `alpha` & `theta`
    for weighting & thresholding.

    ---
    ### Parameters
    `candidate_metrics: pd.DataFrame`: A DataFrame of domain relevance (`"DR"`)
    & domain consensus (`"DC"`), indexed by candidate-bigrams' normalized
    string representations.

    `alpha: float`: Hyperparameter to determine the relative weights of the
    `"DR"` & `"DC"` in the combined candidate-`"score"`. Must be in the closed
    interval `[0.0, 1.0]`.

    `theta: float`: The `"score"`-threshold above which a candidate is
    considered domain terminology. Must be in the half-open interval
    `[0.0, inf)`.

    ---
    ### Returns
    `pd.DataFrame`: A DataFrame, indexed by candidate-bigrams' normalized
    string representations, containing the following columns:
    - `"DR"`: Domain relevance
    - `"DC"`: Domain consensus
    - `"score"`: _terminology_-_score_ based on domain relevance & domain
    consensus, weighted by `alpha`
    - `"is_term"`: Boolean-flag, indicating whether the candidate is at or
    above the _terminology_-_threshold_ `theta`
    """
    if not isinstance(alpha, float):
        if not isinstance(alpha, int):
            logger.error("Invalid type for alpha. Must be a float.")
            return
        alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        logger.error(
            "Invalid value for alpha. Must be in the range [0.0, 1.0]."
        )
        return
    if not isinstance(theta, float):
        if not isinstance(theta, int):
            logger.error("Invalid type for theta. Must be a float.")
            return
        theta = float(theta)
    if not 0.0 <= theta < np.inf:
        logger.error("Invalid value for theta. Must be greater than 0.0.")
        return
    if not isinstance(candidate_metrics, pd.DataFrame):
        logger.error(
            "Invalid type for candidate_metrics. Must be a pd.DataFrame."
        )
        return
    if not all(col in candidate_metrics.columns for col in ["DR", "DC"]):
        logger.error(
            "Missing columns in candidate_metrics. DataFrame must " +
            "contain 'DR' & 'DC'."
        )

    candidate_metrics["score"] = (alpha * candidate_metrics["DR"]) + (
        (1 - alpha) * candidate_metrics["DC"]
    )
    candidate_metrics["is_term"] = candidate_metrics["score"] >= theta
    candidate_metrics = candidate_metrics.sort_values(
        by="score", ascending=False
    )
    n_terms: int = int(candidate_metrics["is_term"].sum())
    perc_candidates: float = round(
        (n_terms / candidate_metrics.shape[0]) * 100, 2
    )
    logger.debug(
        f"Extracted {n_terms} terms ({perc_candidates}% of candidates)."
    )
    return candidate_metrics


def run_experiments(
    candidate_metrics: pd.DataFrame, hyperparams: list[Tuple[float, float]],
    save_experiments: bool = False, gold_terms_filepath: str = None
) -> pd.DataFrame:
    """Run a series of experiments to extract domain terminology using
    different combinations of hyperparameters.

    ---
    ### Parameters
    `candidate_metrics: pd.DataFrame`: A DataFrame of candidate bigrams with
    their respective domain relevance (`"DR"`) & domain consensus (`"DC"`).

    `hyperparams: list[tuple[float, float]]`: A list of tuples, each containing
    two hyperparameters `(alpha, theta)`, where `alpha` determines the relative
    weights of the `"DR"` & `"DC"` in the combined candidate-`"score"`, and
    `theta` is the `"score"`-threshold above which a candidate is considered
    domain terminology.

    `save_experiments: bool`: Whether to save the extracted terms to disk at
    `../output/`. (default: `False`)

    `gold_terms_filepath: str`: The path to the file containing the gold
    standard terms. If not provided, will be loaded from default set.

    ---
    ### Returns
    `pd.DataFrame`: A DataFrame summarizing the results of all experiment-runs.
    Contains the following columns:
    - `"alpha"`: The hyperparameter `alpha`
    - `"theta"`: The hyperparameter `theta`
    - `"precision"`: Precision of the extracted terms
    - `"recall"`: Recall of the extracted terms
    - `"f1"`: (Harmonic) F1-score of the extracted terms
    - `"n_terms"`: Number of extracted terms, given the hyperparameters
    """

    if not isinstance(candidate_metrics, pd.DataFrame):
        logger.error(
            "Invalid type for candidate_metrics. Must be a pd.DataFrame."
        )
        return
    elif not all(col in candidate_metrics.columns for col in ["DR", "DC"]):
        logger.error(
            "Missing columns in candidate_metrics. DataFrame must " +
            "contain 'DR' & 'DC'."
        )
        return
    if not (isinstance(hyperparams, list)
            or not all(isinstance(hp, tuple) for hp in hyperparams)):
        logger.error("Invalid type for hyperparams. Must be a list of tuples.")
        return
    elif not (all(len(hp) == 2 for hp in hyperparams)
              and all(isinstance(hp, float)
                      for params in hyperparams
                      for hp in params)):
        logger.error(
            "Hyperparameters must be tuples of floats (alpha, theta)."
        )
        return

    hyperparams = sorted(hyperparams, key=lambda hp: hp[0])  # sort by alpha
    summary = []
    gold: set[str] = gold_terms(gold_terms_filepath)  # load into memory

    for alpha, theta in hyperparams:
        candidate_metrics = extract_terms(candidate_metrics, alpha, theta)
        terms: pd.Series = candidate_metrics[
            candidate_metrics["is_term"]
        ]["score"]
        summary.append(evaluate_run(set(terms.index), alpha, theta, gold))
        if save_experiments:
            save_terms(alpha, theta, terms, verbose=False)

    summary_df = pd.DataFrame(summary, index=range(1, len(hyperparams) + 1))
    summary_df = summary_df.sort_values(
        by="f1", ascending=False
    )
    summary_str = summary_df.to_string(
        line_width=get_terminal_size().columns, index=True, header=[
            "Alpha", "Theta", "Precision", "Recall", "F1", "# Terms"
        ]
    )
    logger.info(
        f"Summary of {len(hyperparams)} " +
        "experiment " + ("runs" if len(hyperparams) > 1 else "run") +
        f":\n\n{summary_str}\n"
    )
    if save_experiments:
        logger.info(f"Saved experiment results to '\033[3m{OUT_DIR}\033[0m'.")
    return summary_df


### EXAMPLE USAGE ###
if __name__ == "__main__":
    from processing import process_and_vectorize_corpora

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
        domain_sample, ref_sample, n_process=1
    )

    candidate_metrics: pd.DataFrame = compute_dr_dc(
        domain_matrix, ref_matrix, vocab, parallel=False, n_jobs=1
    )

    summary: pd.DataFrame = run_experiments(candidate_metrics, hyperparams=[
        (0.5, 2.0), (0.5, 1.93), (0.7, 1.93), (0.85, 1.37), (0.85, 1.93)
    ])
