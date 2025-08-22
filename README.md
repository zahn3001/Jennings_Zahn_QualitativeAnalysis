# Jennings_Zahn_QualitativeAnalysis
This is the code for the inductive and deductive thematic analysis as described in Evaluating LLM Assisted Qualitative Analysis in Medical Education Research: A Comparison of Human and AI-Generated Thematic Coding.

# Qualitative Analysis with OpenAI

This repository contains Python scripts to support qualitative coding analysis (both inductive and deductive) using the OpenAI API, guided by Braun & Clarke’s thematic analysis framework. The scripts are designed to be user-friendly and configurable so that researchers can adapt them to their own transcripts and analysis workflows.

## Repository Structure


qualitative-analysis/
├─ README.md
├─ requirements.txt
├─ qualitative-analysis/
│  ├─ Step1_Familiarization_JZQA.py   # Inductive coding - Step 1
│  ├─ Step2_InitialCoding.py          # Inductive coding - Step 2
│  ├─ Step3_CodebookDevelopment.py    # Inductive coding - Step 3
│  ├─ Deductive_Coding.py             # Deductive coding analysis
├─ data/
│  ├─ inputs/                         # Place your input files here
│  └─ samples/                        # Small example transcripts for testing
├─ outputs/
│  ├─ inductive/                      # Saved results for inductive steps
│  └─ deductive/                      # Saved results for deductive coding
├─ configs/
│  └─ example.env                     # Shows how to set environment variables
└─ docs/
   ├─ quickstart.md
   └─ methods.md

## Requirements

- Python 3.10+
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- pandas

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
