# LLM Memorization Paper Implementation

This repository contains an implementation for analyzing and evaluating memorization behavior in Large Language Models (LLMs), with a focus on encoder-decoder architectures like BERT and BART.

## Overview

This project investigates memorization properties in transformer-based language models through empirical analysis. The implementation provides tools to evaluate how models memorize training data and examines the differences between encoder-only, decoder-only, and encoder-decoder architectures.

## Repository Structure

- `experimentMemorization.ipynb` - Main notebook containing memorization experiments and analysis
- `BertBartAnalysiss.ipynb` - Comparative analysis of BERT and BART architectures for memorization studies
- `ResultsPaser.py` - Utility script for parsing and processing experimental results
- `encoderResultExtractor.py` - Specialized extractor for encoder model results
- `evaluation_results.json` - Compiled evaluation metrics from experiments
- `EncoderResults.json` - Detailed results specifically for encoder models

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required libraries (transformers, torch, numpy, etc.)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rohmish2/LLM-Memorization-Paper-Implementation.git
cd LLM-Memorization-Paper-Implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: If a requirements.txt file is not present, install the following commonly used packages:
```bash
pip install torch transformers numpy pandas jupyter
```

## Usage

### Running Experiments

1. Open the main experiment notebook:
```bash
jupyter notebook experimentMemorization.ipynb
```

2. For BERT/BART comparative analysis:
```bash
jupyter notebook BertBartAnalysiss.ipynb
```

### Processing Results

To parse and analyze experimental results:
```bash
python ResultsPaser.py
```

To extract encoder-specific results:
```bash
python encoderResultExtractor.py
```

## Methodology

The implementation focuses on:

- Measuring memorization in pre-trained language models
- Comparing memorization patterns across different architectures (BERT vs BART)
- Analyzing encoder-decoder model behavior
- Extracting and quantifying memorized sequences

## Results

Results are stored in JSON format for reproducibility and further analysis:
- `evaluation_results.json` - General evaluation metrics
- `EncoderResults.json` - Encoder-specific measurements

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests.

