#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Mon, 10.03.25                                                     #
# ========================================================================== #

"""CLI for terminology extraction."""

import click
from itertools import product
from json import JSONDecodeError, loads
from math import inf
from multiprocessing import cpu_count
import os
from pandas import DataFrame
import sys
from time import time

sys.path.insert(0, os.path.abspath(  # add src to path
    os.path.join(os.path.dirname(__file__), '.')
))
OUT_DIR = os.path.abspath(  # set output directory
    os.path.join("..", os.path.dirname(__file__), "output")
)
from computations import compute_dr_dc, run_experiments  # noqa: E402
from processing import process_and_vectorize_corpora  # noqa: E402
from utils import load_reuters, parse_bibtex_abstracts  # noqa: E402


@click.group(**{"help": str(click.style(
    "CLI for terminology extraction.",
    bold=True, underline=True, bg="white", fg="black"
))})
def cli():
    """CLI for terminology extraction."""
    pass


@cli.command(**{
    "help": str(
        str(click.style("Perform end-to-end terminology extraction:",
                        bold=True, bg="white", fg="black")) +
        "\n\n(1.) Parses domain abstracts from a BibTex-file, " +
        "\n\n(2.) processes and vectorizes domain & reference texts (Reuters),"
        + "\n\n(3.) extracts candidate bigrams (word-like pairs of " +
        "ADJ/NN/PROPN + NN/PROPN) from domain texts," +
        "\n\n(4.) computes domain relevance and domain consensus scores for " +
        "each candidate and" +
        "\n\n(5.) applies hyperparameters for weighting and extracting " +
        "domain terminology."
    )
})
@click.option(
    "--domain_bibtex", "-d", required=True,
    help="Path to a BibTex-file containing abstracts.",
    type=click.Path(
        exists=True, dir_okay=False, readable=True, resolve_path=True
    )
)
@click.option(
    "--alpha", "-a", required=True, type=click.FloatRange(0.0, 1.0),
    help=str(
        "The weight ratio of \033[3mdomain relevance\033[0m and " +
        "\033[3mdomain consensus\033[0m, used to compute the " +
        "terminology-score of a candidate. Provide at least one value.\n" +
        "(If multiple values are provided, the cartesian product of " +
        "alpha and theta will be used as hyperparameters.)"
    ), multiple=True
)
@click.option(
    "--theta", "-t", required=True, multiple=True,
    type=click.FloatRange(0.0, inf, max_open=True),
    help=str(
        "The \033[3mterminology-score\033[0m threshold, at or above which a " +
        "candidate is considered domain terminology. Provide at least one " +
        "value.\n" +
        "(If multiple values are provided, the cartesian product of " +
        "alpha and theta will be used as hyperparameters.)"
    )
)
@click.option(
    "--save", type=bool, default=False, required=False,
    help=str(
        "Flag to indicate whether extracted terms should be saved to " +
        f"'{OUT_DIR}'."
    )
)
@click.option(
    "--gold_filepath", type=click.Path(
        exists=True, dir_okay=False, readable=True, resolve_path=True
    ), required=False, default=None, show_default=False,
    help="Path to a TXT-file containing a single gold-standard term per line.",
)
@click.option(
    "--process-kwargs", type=str, required=False,
    default='{"english_only": "True", "spacy_model": "en_core_web_md", "batch_size": "1000", "n_process": "-1"}',  # noqa: E501
    help="JSON-formatted string with additional kwargs for processing.",
    show_default=True
)
@click.option(
    "--compute-kwargs", type=str, required=False,
    default='{"parallel": "True", "n_jobs": "-1"}',
    help="JSON-formatted string with additional kwargs for computation.",
    show_default=True
)
def extract(
    domain_bibtex, alpha, theta, save, gold_filepath,
    process_kwargs, compute_kwargs
):
    """Perform end-to-end terminology extraction."""
    start = time()

    # create hyperparameter-combinations
    if (n_a := len(alpha)) > 1 and (n_t := len(theta)) > 1:
        if n_a != n_t:
            hyperparams = list(product(alpha, theta))
        else:
            hyperparams = list(zip(alpha, theta))

    # parse kwargs
    try:
        proc_kwargs = loads(process_kwargs, )
        en_only: bool = bool(proc_kwargs.get("english_only", True))
        spacy_model: str = str(
            proc_kwargs.get("spacy_model", "en_core_web_md")
        )
        batch_size: int = int(proc_kwargs.get("batch_size", 1000))
        n_process: int = int(proc_kwargs.get("n_process", cpu_count() - 2))
    except JSONDecodeError as e:
        click.echo(f"Failed to parse processing kwargs. {e}")
        try:
            click.confirm(
                "Y: continue with default values, N: retry with new input",
                abort=True
            )
        except click.Abort:  # user wants to retry
            return
    try:
        comp_kwargs = loads(compute_kwargs)
        parallel: bool = bool(comp_kwargs.get("parallel", True))
        n_jobs: int = bool(comp_kwargs.get("n_jobs", -1))
    except JSONDecodeError as e:
        click.echo(f"Failed to parse computation kwargs. {e}")
        try:
            click.confirm(
                "Y: continue with default values, N: retry with new input",
                abort=True
            )
        except click.Abort:
            return

    # load domain & reference texts
    domain_abstracts: list[str] = parse_bibtex_abstracts(
        domain_bibtex, english_only=en_only
    )
    reference_texts: list[str] = load_reuters()

    # preprocess & vectorize texts
    domain_matrix, ref_matrix, vocab = process_and_vectorize_corpora(
        domain_abstracts, reference_texts,
        spacy_model=spacy_model, batch_size=batch_size, n_process=n_process
    )

    # compute domain relevance and domain consensus
    candidate_metrics: DataFrame = compute_dr_dc(
        domain_matrix, ref_matrix, vocab,
        parallel=parallel, n_jobs=n_jobs
    )

    # compute terminology scores & extract terms with given hyperparameters
    run_experiments(
        candidate_metrics=candidate_metrics, hyperparams=hyperparams,
        save_experiments=save, gold_terms_filepath=gold_filepath
    )
    m, s = [int(i) for i in divmod(time() - start, 60)]
    click.secho(
        f"Done! Extraction took {m}m {s}s.", fg="cyan", bold=True
    )


if __name__ == "__main__":
    cli()
