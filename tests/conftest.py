#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Fri, 07.03.25                                                     #
# ========================================================================== #

"""Test fixtures"""

import numpy as np
import os
import pandas as pd
import pytest
from scipy import sparse
from spacy.tokens import Doc
import sys
from typing import Generator

# import src package
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
))
from src.processing import SpacyProcessor


@pytest.fixture(scope="module")
def domain_sample() -> list[str]:
    """List of sample domain documents."""
    return [
        "In this paper, we introduce the concept of Semantic Masking, where semantically coherent surrounding text (the haystack) interferes with the retrieval and comprehension of specific information (the needle) embedded within it. We propose the Needle-in-a-Haystack-QA Test, an evaluation pipeline that assesses LLMs' long-text capabilities through question answering, explicitly accounting for the Semantic Masking effect. We conduct experiments to demonstrate that Semantic Masking significantly impacts LLM performance more than text length does. By accounting for Semantic Masking, we provide a more accurate assessment of LLMs' true proficiency in utilizing extended contexts, paving the way for future research to develop models that are not only capable of handling longer inputs but are also adept at navigating complex semantic landscapes.",  # noqa: E501
        "Nowadays, AI is present in all our activities. This pervasive presence is perceived as a threat by many category of users that might be substituted by their AI counterpart. While the potential of AI in handling repetitive tasks is clear, the potentials of its creativeness is still misunderstood. We believe that understanding this aspects of AI can transform a threat into an opportunity. This paper is a first attempt to provide a measurable definition of creativeness. We applied our definition to AI and human generated texts, proving the viability of the proposed approach. Our preliminary experiments show that human texts are more creative.",  # noqa: E501
        "While MSA and some dialects of Arabic have been extensively studied in NLP, Middle Arabic is still very much unknown to the field. However, Middle Arabic holds issues that are still not covered: it is characterized by variation since it mixes standard features, colloquial ones, as well as features that belong to neither of the two. Here, we introduce a methodology to identify, extract and rank variations of 13 manually retrieved formulas. Those formulas come from the nine first booklets of S ̄IRAT AL-MALIK AL-Z. ̄AHIR BAYBAR S., a corpus of Damascene popular literature written in Middle Arabic and composed of 53,843 sentences. In total, we ranked 20, sequences according to their similarity with the original formulas on multiple linguistic layers. We noticed that the variations in these formulas occur in a lexical, morphological and graphical level, but in opposition, the semantic and syntactic levels remain strictly invariable.",  # noqa: E501
        "In this paper, we introduce the concept of Semantic Masking, where semantically coherent surrounding text (the haystack) interferes with the retrieval and comprehension of specific information (the needle) embedded within it. We propose the Needle-in-a-Haystack-QA Test, an evaluation pipeline that assesses LLMs' long-text capabilities through question answering, explicitly accounting for the Semantic Masking effect. We conduct experiments to demonstrate that Semantic Masking significantly impacts LLM performance more than text length does. By accounting for Semantic Masking, we provide a more accurate assessment of LLMs' true proficiency in utilizing extended contexts, paving the way for future research to develop models that are not only capable of handling longer inputs but are also adept at navigating complex semantic landscapes.",  # noqa: E501
    ]


@pytest.fixture(scope="module")
def ref_sample() -> list[str]:
    """List of reference documents.

    (ref_sample[-1] is a duplicate of domain_sample[1])
    """
    return [
        "CHINA DAILY SAYS VERMIN EAT 7-12 PCT GRAIN STOCKS\n  A survey of 19 provinces and seven cities\n  showed vermin consume between seven and 12 pct of China's grain\n  stocks, the China Daily said.\n      It also said that each year 1.575 mln tonnes, or 25 pct, of\n  China's fruit output are left to rot, and 2.1 mln tonnes, or up\n  to 30 pct, of its vegetables. The paper blamed the waste on\n  inadequate storage and bad preservation methods.\n      It said the government had launched a national programme to\n  reduce waste, calling for improved technology in storage and\n  preservation, and greater production of additives. The paper\n  gave no further details.\n  \n\n",  # noqa: E501
        "THAI TRADE DEFICIT WIDENS IN FIRST QUARTER\n  Thailand's trade deficit widened to 4.5\n  billion baht in the first quarter of 1987 from 2.1 billion a\n  year ago, the Business Economics Department said.\n      It said Janunary/March imports rose to 65.1 billion baht\n  from 58.7 billion. Thailand's improved business climate this\n  year resulted in a 27 pct increase in imports of raw materials\n  and semi-finished products.\n      The country's oil import bill, however, fell 23 pct in the\n  first quarter due to lower oil prices.\n      The department said first quarter exports expanded to 60.6\n  billion baht from 56.6 billion.\n      Export growth was smaller than expected due to lower\n  earnings from many key commodities including rice whose\n  earnings declined 18 pct, maize 66 pct, sugar 45 pct, tin 26\n  pct and canned pineapples seven pct.\n      Products registering high export growth were jewellery up\n  64 pct, clothing 57 pct and rubber 35 pct.\n  \n\n",  # noqa: E501
        "Nowadays, AI is present in all our activities. This pervasive presence is perceived as a threat by many category of users that might be substituted by their AI counterpart. While the potential of AI in handling repetitive tasks is clear, the potentials of its creativeness is still misunderstood. We believe that understanding this aspects of AI can transform a threat into an opportunity. This paper is a first attempt to provide a measurable definition of creativeness. We applied our definition to AI and human generated texts, proving the viability of the proposed approach. Our preliminary experiments show that human texts are more creative."  # noqa: E501
    ]


@pytest.fixture(scope="module")
def domain_matrix() -> sparse.csr_matrix:
    """Vectorized (sparse) domain matrix."""
    return sparse.csr_matrix([
        [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 0, 1, 4, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
         1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 1,
         0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 0, 1, 4, 1, 0, 0, 1, 1, 0, 0]
    ])


@pytest.fixture(scope="module")
def ref_matrix() -> sparse.csr_matrix:
    """Vectorized (sparse) reference matrix."""
    return sparse.csr_matrix([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
         1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])


@pytest.fixture(scope="module")
def vocab() -> dict[str, int]:
    """Mapping of candidates to column indices."""
    return {
        "accurate assessment": 0,
        "ai counterpart": 1,
        "colloquial ones": 2,
        "evaluation pipeline": 3,
        "extended contexts": 4,
        "first attempt": 5,
        "first booklets": 6,
        "future research": 7,
        "graphical level": 8,
        "human texts": 9,
        "linguistic layers": 10,
        "llm performance": 11,
        "long-text capabilities": 12,
        "longer inputs": 13,
        "many category": 14,
        "masking effect": 15,
        "measurable definition": 16,
        "middle arabic": 17,
        "needle-in-a-haystack-qa test": 18,
        "original formulas": 19,
        "pervasive presence": 20,
        "popular literature": 21,
        "preliminary experiments": 22,
        "question answering": 23,
        "rank variations": 24,
        "repetitive tasks": 25,
        "s ̄irat": 26,
        "semantic landscapes": 27,
        "semantic masking": 28,
        "specific information": 29,
        "standard features": 30,
        "syntactic levels": 31,
        "text length": 32,
        "true proficiency": 33,
        "̄ahir baybar": 34,
        "̄irat al-malik": 35
    }


@pytest.fixture(scope="module")
def domain_cf() -> np.ndarray:
    """Collection frequencies of candidates in `domain_sample`."""
    return np.array([
        2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 3, 2, 1, 1, 1,
        1, 2, 1, 1, 1, 2, 8, 2, 1, 1, 2, 2, 1, 1
    ])


@pytest.fixture(scope="module")
def ref_cf() -> np.ndarray:
    """Collection frequencies of candidates in `ref_sample`."""
    return np.array([
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])


@pytest.fixture(scope="module")
def domain_rel_freqs() -> np.ndarray:
    """Relative frequencies of candidates in `domain_sample`."""
    return np.array([
        0.03389831, 0.01694915, 0.01694915, 0.03389831, 0.03389831,
        0.01694915, 0.01694915, 0.03389831, 0.01694915, 0.01694915,
        0.01694915, 0.03389831, 0.03389831, 0.03389831, 0.01694915,
        0.03389831, 0.01694915, 0.05084746, 0.03389831, 0.01694915,
        0.01694915, 0.01694915, 0.01694915, 0.03389831, 0.01694915,
        0.01694915, 0.01694915, 0.03389831, 0.13559322, 0.03389831,
        0.01694915, 0.01694915, 0.03389831, 0.03389831, 0.01694915,
        0.01694915
    ])


@pytest.fixture(scope="module")
def ref_rel_freqs() -> np.ndarray:
    """Relative frequencies of candidates in `ref_sample`."""
    return np.array([
        0.0, 0.125, 0.0, 0.0, 0.0, 0.125, 0.0, 0.0, 0.0,
        0.125, 0.0, 0.0, 0.0, 0.0, 0.125, 0.0, 0.125, 0.0,
        0.0, 0.0, 0.125, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
    ])


@pytest.fixture(scope="module")
def candidates_domain_relevance(vocab) -> pd.Series:
    """Domain relevance of candidates in `domain_sample`, defined as:

    ```python
    dr[i] = domain_rel_freqs[i] / (domain_rel_freqs[i] + ref_rel_freqs[i])
    ```
    """
    dr = np.array([
        1.0, 0.11940299, 1.0, 1.0, 1.0, 0.11940299, 1.0, 1.0, 1.0, 0.11940299,
        1.0, 1.0, 1.0, 1.0, 0.11940299, 1.0, 0.11940299, 1.0, 1.0, 1.0,
        0.11940299, 1.0, 0.11940299, 1.0, 1.0, 0.11940299, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ])
    return pd.Series(dr, index=vocab.keys(), name="DR", dtype=np.float64)


@pytest.fixture(scope="function")
def candidate_distributions() -> Generator[np.ndarray, None, None]:
    """Probability distributions of candidates over each document in
    `domain_sample`. Each row corresponds to a document, each column to a
    candidate."""
    yield (np.array(dist) for dist in [
        [0.5, 0.5],
        [0.5],
        [0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5],
        [0.5],
        [0.5, 0.125],
        [0.5],
        [0.5],
        [0.5],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0],
        [1.0, 1.0],
        [1.0],
        [3.0],
        [1.0, 0.33333333],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0, 1.0],
        [1.0],
        [1.0],
        [0.5],
        [0.5, 0.5],
        [2.0, 2.0],
        [0.5, 0.5],
        [0.5],
        [0.5],
        [0.5, 0.5],
        [0.125, 0.5],
        [0.5],
        [0.5]
    ])


@pytest.fixture(scope="module")
def candidates_domain_consensus() -> pd.Series:
    """Domain consensus of candidates in `domain_sample`, computed as:

    ```python
    dc[i] := entropy(domain_rel_freqs[i])
    entropy(domain_rel_freqs[i]) = -sum(
        domain_rel_freqs[i] * log(domain_rel_freqs[i])  # log-base = e
    )
    ```
    """
    dc = np.array([
        1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.72192809, 0.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 0.0, 0.81127812, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.72192809, 0.0, 0.0
    ])
    idx = [
        "accurate assessment",
        "ai counterpart",
        "colloquial ones",
        "evaluation pipeline",
        "extended contexts",
        "first attempt",
        "first booklets",
        "future research",
        "graphical level",
        "human texts",
        "linguistic layers",
        "llm performance",
        "long-text capabilities",
        "longer inputs",
        "many category",
        "masking effect",
        "measurable definition",
        "middle arabic",
        "needle-in-a-haystack-qa test",
        "original formulas",
        "pervasive presence",
        "popular literature",
        "preliminary experiments",
        "question answering",
        "rank variations",
        "repetitive tasks",
        "s ̄irat",
        "semantic landscapes",
        "semantic masking",
        "specific information",
        "standard features",
        "syntactic levels",
        "text length",
        "true proficiency",
        "̄ahir baybar",
        "̄irat al-malik"
    ]
    return pd.Series(dc, index=idx, name="DC", dtype=np.float64)


@pytest.fixture(scope="function")
def candidate_scores(
    candidates_domain_relevance, candidates_domain_consensus
) -> np.ndarray:
    """Expected terminology-scores (in descending order) of candidates in
    `domain_sample`, with `alpha = 0.65` and `theta = 0.1`."""
    return np.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.93394734, 0.90267483, 0.90267483,
        0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
        0.65,
        0.07761194, 0.07761194, 0.07761194, 0.07761194, 0.07761194, 0.07761194,
        0.07761194, 0.07761194
    ])


@pytest.fixture(scope="module")
def sample_terms() -> set[str]:
    """Set of sample terminology, extracted from `domain_sample`."""
    return {
        "accurate assessment",
        "evaluation pipeline",
        "extended contexts",
        "llm performance",
        "long-text capabilities",
        "longer inputs",
        "masking effect",
        "question answering",
        "semantic landscapes",
        "semantic masking",
        "specific information",
        "text length"
    }


@pytest.fixture(scope="module")
def sample_gold_terms() -> set[str]:
    """Set of gold terminology as a subset of `sample_terms`."""
    return {
        "accurate assessment",
        "evaluation pipeline",
        "extended contexts",
        "llm performance",
        "long-text capabilities",
        "longer inputs"
    }


@pytest.fixture(scope="module")
def raw_bibtex() -> str:
    """A raw BibTeX string to test parsing."""
    return """
@proceedings{wraicogs-ws-2025-1,
    title = "Proceedings of the First Workshop on Writing Aids at the Crossroads of AI, Cognitive Science and NLP (WRAICOGS 2025)",
    editor = "Zock, Michael  and
      Inui, Kentaro  and
      Yuan, Zheng",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2025.wraicogs-1.0/"
}
@inproceedings{buhnila-etal-2025-chain,
    title = "Chain-of-{M}eta{W}riting: Linguistic and Textual Analysis of How Small Language Models Write Young Students Texts",
    author = "Buhnila, Ioana  and
      Cislaru, Georgeta  and
      Todirascu, Amalia",
    editor = "Zock, Michael  and
      Inui, Kentaro  and
      Yuan, Zheng",
    booktitle = "Proceedings of the First Workshop on Writing Aids at the Crossroads of AI, Cognitive Science and NLP (WRAICOGS 2025)",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2025.wraicogs-1.1/",
    pages = "1--15",
    abstract = "Large Language Models (LLMs) have been used to generate texts in response to different writing tasks: reports, essays, story telling. However, language models do not have a metarepresentation of the text writing process, nor inherent communication learning needs, comparable to those of young human students. This paper introduces a fine-grained linguistic and textual analysis of multilingual Small Language Models' (SLMs) writing. With our method, Chain-of-MetaWriting, SLMs can imitate some steps of the human writing process, such as planning and evaluation. We mainly focused on short story and essay writing tasks in French for schoolchildren and undergraduate students respectively. Our results show that SLMs encounter difficulties in assisting young students on sensitive topics such as violence in the schoolyard, and they sometimes use words too complex for the target audience. In particular, the output is quite different from the human produced texts in term of text cohesion and coherence regarding temporal connectors, topic progression, reference."
}
@inproceedings{cardon-etal-2024-apport,
    title = "Apport de la structure de tours {\`a} l`identification automatique de genre textuel: un corpus annot{\'e} de sites web de tourisme en fran{\c{c}}ais",
    author = "Cardon, Remi  and
      Tran Hanh Pham, Trang  and
      Zakhia Doueihi, Julien  and
      Fran{\c{c}}ois, Thomas",
    editor = "Balaguer, Mathieu  and
      Bendahman, Nihed  and
      Ho-dac, Lydia-Mai  and
      Mauclair, Julie  and
      G Moreno, Jose  and
      Pinquier, Julien",
    booktitle = "Actes de la 31{\`e}me Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles, volume 2 : traductions d`articles publi{\`e}s",
    month = "7",
    year = "2024",
    address = "Toulouse, France",
    publisher = "ATALA and AFPC",
    url = "https://aclanthology.org/2024.jeptalnrecital-trad.1/",
    pages = "1--1",
    language = "fra",
    abstract = "Ce travail {\'e}tudie la contribution de la structure de tours {\`a} l`identification automatique de genres textuels. Ce concept {--} bien connu dansle domaine de l`analyse de genre {--} semble {\^e}tre peu exploit{\'e} dans l`identification automatique du genre. Nous d{\'e}crivons la collecte d`un corpus de sites web francophones relevant du domaine du tourisme et le processus d`annotation avec les informations de tours. Nous menons des exp{\'e}riences d`identification automatique du genre de texte avec notre corpus. Nos r{\'e}sultats montrent qu`ajouter l`information sur la structure de tours dans un mod{\`e}le am{\'e}liore ses performances pour l`identification automatique du genre, tout en r{\'e}duisant le volume de donn{\'e}es n{\'e}cessaire et le besoin en ressource de calcul."
}
@inproceedings{fugeng-etal-2024-construction,
    title = "Construction of {CFSP} Model Based on Non-Finetuning Large Language Model",
    author = "Fugeng, Huang  and
      Zhongbin, Guo  and
      Wenting, Li  and
      Haibo, Cheng",
    editor = "Lin, Hongfei  and
      Tan, Hongye  and
      Li, Bin",
    booktitle = "Proceedings of the 23rd Chinese National Conference on Computational Linguistics (Volume 3: Evaluations)",
    month = jul,
    year = "2024",
    address = "Taiyuan, China",
    publisher = "Chinese Information Processing Society of China",
    url = "https://aclanthology.org/2024.ccl-3.1/",
    pages = "1--9",
    language = "eng",
    abstract = "{\textquotedblleft}Chinese Frame Semantic Parsing (CFSP) is an important task in the field of Chinese Natural Language Processing(NLP). Its goal is to extract the frame semantic structure from the sentence and realize the deep understanding of the events or situations involved in the sentence. This paper mainly studies the application of Large Language Model (LLM) for reasoning through Prompt Engineering without fine-tuning the model, and completes three subtasks of Chinese Framework Semantic Parsing tasks: frame identification, argument Identification and role identification. This paper proposes a Retrieval Augmented Generation (RAG) method for target words, and constructs more refined sample Few-Shot method. We achieved the second place on the B rankings in the open track in the {\textquotedblleft}CCL2024-Eval The Second Chinese Frame Semantic Parsing{\textquotedblright}competition*.{\textquotedblright}"
}
"""


@pytest.fixture(scope="module")
def parsed_bibtex_en() -> list[str]:
    """A list of **English** _abstracts_, contained in `raw_bibtex`."""
    return [
        "Large Language Models (LLMs) have been used to generate texts in response to different writing tasks: reports, essays, story telling. However, language models do not have a metarepresentation of the text writing process, nor inherent communication learning needs, comparable to those of young human students. This paper introduces a fine-grained linguistic and textual analysis of multilingual Small Language Models' (SLMs) writing. With our method, Chain-of-MetaWriting, SLMs can imitate some steps of the human writing process, such as planning and evaluation. We mainly focused on short story and essay writing tasks in French for schoolchildren and undergraduate students respectively. Our results show that SLMs encounter difficulties in assisting young students on sensitive topics such as violence in the schoolyard, and they sometimes use words too complex for the target audience. In particular, the output is quite different from the human produced texts in term of text cohesion and coherence regarding temporal connectors, topic progression, reference.",  # noqa: E501
        ' extquotedblleftChinese Frame Semantic Parsing (CFSP) is an important task in the field of Chinese Natural Language Processing(NLP). Its goal is to extract the frame semantic structure from the sentence and realize the deep understanding of the events or situations involved in the sentence. This paper mainly studies the application of Large Language Model (LLM) for reasoning through Prompt Engineering without fine-tuning the model, and completes three subtasks of Chinese Framework Semantic Parsing tasks: frame identification, argument Identification and role identification. This paper proposes a Retrieval Augmented Generation (RAG) method for target words, and constructs more refined sample Few-Shot method. We achieved the second place on the B rankings in the open track in the extquotedblleftCCL2024-Eval The Second Chinese Frame Semantic Parsing extquotedblrightcompetition*. extquotedblright'  # noqa: E501
    ]


@pytest.fixture(scope="module")
def parsed_bibtex_all() -> list[str]:
    """A list of **all** _abstracts_, contained in `raw_bibtex`."""
    return [
        "Large Language Models (LLMs) have been used to generate texts in response to different writing tasks: reports, essays, story telling. However, language models do not have a metarepresentation of the text writing process, nor inherent communication learning needs, comparable to those of young human students. This paper introduces a fine-grained linguistic and textual analysis of multilingual Small Language Models' (SLMs) writing. With our method, Chain-of-MetaWriting, SLMs can imitate some steps of the human writing process, such as planning and evaluation. We mainly focused on short story and essay writing tasks in French for schoolchildren and undergraduate students respectively. Our results show that SLMs encounter difficulties in assisting young students on sensitive topics such as violence in the schoolyard, and they sometimes use words too complex for the target audience. In particular, the output is quite different from the human produced texts in term of text cohesion and coherence regarding temporal connectors, topic progression, reference.",  # noqa: E501
        "Ce travail 'etudie la contribution de la structure de tours à l`identification automatique de genres textuels. Ce concept -- bien connu dansle domaine de l`analyse de genre -- semble être peu exploit'e dans l`identification automatique du genre. Nous d'ecrivons la collecte d`un corpus de sites web francophones relevant du domaine du tourisme et le processus d`annotation avec les informations de tours. Nous menons des exp'eriences d`identification automatique du genre de texte avec notre corpus. Nos r'esultats montrent qu`ajouter l`information sur la structure de tours dans un modł̀e am'eliore ses performances pour l`identification automatique du genre, tout en r'eduisant le volume de donn'ees n'ecessaire et le besoin en ressource de calcul.",  # noqa: E501
        ' extquotedblleftChinese Frame Semantic Parsing (CFSP) is an important task in the field of Chinese Natural Language Processing(NLP). Its goal is to extract the frame semantic structure from the sentence and realize the deep understanding of the events or situations involved in the sentence. This paper mainly studies the application of Large Language Model (LLM) for reasoning through Prompt Engineering without fine-tuning the model, and completes three subtasks of Chinese Framework Semantic Parsing tasks: frame identification, argument Identification and role identification. This paper proposes a Retrieval Augmented Generation (RAG) method for target words, and constructs more refined sample Few-Shot method. We achieved the second place on the B rankings in the open track in the extquotedblleftCCL2024-Eval The Second Chinese Frame Semantic Parsing extquotedblrightcompetition*. extquotedblright'  # noqa: E501
    ]


@pytest.fixture(scope="module")
def hyphenated_multiword_text() -> str:
    """A raw text containing hyphenated multi-word terms."""
    return "In this paper, we introduce the concept of Semantic Masking, where semantically coherent surrounding text (the haystack) interferes with the retrieval and comprehension of specific information (the needle) embedded within it. We propose the Needle-in-a-Haystack-QA Test, an evaluation pipeline that assesses LLMs' long-text capabilities through question answering, explicitly accounting for the Semantic Masking effect. We conduct experiments to demonstrate that Semantic Masking significantly impacts LLM performance more than text length does. By accounting for Semantic Masking, we provide a more accurate assessment of LLMs' true proficiency in utilizing extended contexts, paving the way for future research to develop models that are not only capable of handling longer inputs but are also adept at navigating complex semantic landscapes."  # noqa: E501


@pytest.fixture(scope="module")
def spacy_processor() -> SpacyProcessor:
    """A `SpacyProcessor`-instance with default settings."""
    return SpacyProcessor("en_core_web_md")


@pytest.fixture(scope="module")
def sample_doc(spacy_processor, hyphenated_multiword_text) -> Doc:
    """A `spacy.tokens.Doc`, processed with `SpacyProcessor`.
    Custom Doc._.candidate_counts-extension is removed.
    """
    doc = spacy_processor.spacy(hyphenated_multiword_text)
    doc.remove_extension("candidate_counts")
    return doc


@pytest.fixture(scope="module")
def sample_doc_counts() -> dict[str, int]:
    """A dictionary of counts of normalized candidate bigrams in `sample_doc`.
    """
    return {
        "semantic masking": 4,
        "specific information": 1,
        "needle-in-a-haystack-qa test": 1,
        "evaluation pipeline": 1,
        "long-text capabilities": 1,
        "question answering": 1,
        "masking effect": 1,
        "llm performance": 1,
        "text length": 1,
        "accurate assessment": 1,
        "true proficiency": 1,
        "extended contexts": 1,
        "future research": 1,
        "longer inputs": 1,
        "semantic landscapes": 1
    }
