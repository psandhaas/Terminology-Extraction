# Terminology Extraction from Domain Corpus

## Overview
This project implements a pipeline for extracting domain-specific terminology from a corpus of scientific abstracts using statistical methods such as Domain Relevance (DR) and Domain Consensus (DC). The project is designed to identify terms that are both frequent and consistent within a domain corpus compared to a reference corpus.

### Project structure

```bash
Terminology-Extraction
├── data/
|   └── gold_terminology_abstracts.txt      # required for evaluation
├── output/                             # results directory
|   ├── exp_1.tsv
|   └── ...
├── src/                                # source code
|   ├── main.py                             # CLI entry point
|   ├── processing.py                       # preprocessing & vectorization
|   ├── computations.py                     # metric calculations & extraction logic
|   └── utils.py
├── tests/                              # unit tests
|   ├── __init__.py
|   ├── conftests.py                        # fixtures
|   ├── test_computations.py                # tests for extraction logic
|   └── test_processing.py                  # tests for preprocessing pipeline
└── README.md                           # documentation
└── requirements.txt                    # pip-requirements
└── requirements_conda.txt              # conda-requirements
```

---

## Features
- Parsing abstracts of scientific publications from BibTeX file
- Corpus preprocessing & vectorization
- Candidate-bigram selection based on token-POS tags
- Computation of DR & DC metrics
- Extraction of domain terminology from set of candidate-bigrams based on hyperparameters

## Requirements
- Python >=3.9
- Libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `spacy`, `click`

## Installation
1. Clone the repository:

```bash
git clone https://github.com/psandhaas/terminology-extraction.git
cd terminology-extraction
```
2. Install dependencies:
   - using pip
    ```bash
   pip install -r requirements.txt
   ```
   - or using conda:
    ```bash
    conda create -n myenv --file requirements_conda.txt
    ```
3. Download the default spaCy model (or provide a different one):
```bash
python -m spacy download en_core_web_md
```

---

## Usage
Run the end-to-end terminology extraction pipeline using the CLI:
```bash
python main.py extract -d <path/to/bibtex.bib> -a 0.5 -t 2.0
```

### Parameters
- `--domain_bibtex`: Path to a BibTeX file containing abstracts.
- `--alpha`: Hyperparameter to weigh relative contributions of DR and DC values of candidates to their terminology scores.
- `--theta`: Hyperparameter that is used as a threshold for terminology scores of candidates. Any candidates with a weighted terminology score equal or greater than theta are extracted.

### Options
- `--save`: Whether to save extracted terms to the `./output/` directory.
- `--help`: Show CLI-help message and exit.
- `--gold_filepath`: Path to a TXT-file containing a single gold-standard term per line.

### Output
The extracted terms are saved as CSV files in the `./output/` directory.

---

## Tests
Run tests from the terminal:
```bash
python -m pytest
```

---

## Background

### Definitions

$$
TerminologyScore = \alpha \cdot DR + (\alpha - 1) \cdot DC
$$
$$
Terminology = \{TerminologyScore(t) \gt \theta : t \in Candidates_{domain} | Candidates_{domain} \subset \Sigma_{domain}\}
$$

### Literature
- Paola Velardi, Paolo Fabriani, and Michele Missikoff. 2001. Using text processing techniques to automatically enrich a domain ontology. In *Proceedings of the international conference on Formal Ontology in Information Systems* - Volume 2001 (FOIS '01). Association for Computing Machinery, New York, NY, USA, 270–284. [https://doi.org/10.1145/505168.505194](https://doi.org/10.1145/505168.505194)
- Wendt, M., Buscher, C., Herta, C., Gerlach, M., Messner, M., Kemmerer, S., & Tietze, W. (2009, September). Extracting domain terminologies from the World Wide Web. In *Web as Corpus Workshop (WAC5)* (p. 79). [PDF](https://www.sigwac.org.uk/raw-attachment/wiki/WAC5/WAC5_proceedings.pdf#page=79)
