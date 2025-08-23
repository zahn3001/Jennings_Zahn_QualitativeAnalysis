# Jennings_Zahn_QualitativeAnalysis
This is the code for the inductive and deductive thematic analysis as described in Evaluating LLM Assisted Qualitative Analysis in Medical Education Research: A Comparison of Human and AI-Generated Thematic Coding.

# Qualitative Analysis with OpenAI

This repository contains Python scripts to support qualitative coding analysis (both inductive and deductive) using the OpenAI API, guided by Braun & Clarke’s thematic analysis framework. The scripts are designed to be user-friendly and configurable so that researchers can adapt them to their own transcripts and analysis workflows.

## Repository Structure


qualitative-analysis/
├─ README.md
├─ requirements.txt
├─ qualitative-analysis/
│  ├─ Step1_Familiarization_JZQA.py
│  ├─ Step2_Coding_JZQA.py      
│  ├─ Step3_FinalCodebook_JZQA.py
├─ data/
│  ├─ inputs/
│  └─ samples/
├─ outputs/
│  ├─ inductive/
│  └─ deductive/
├─ configs/
│  └─ example.env
└─ docs/
   ├─ quickstart.md
   └─ methods.md

## Requirements

- Python 3.10+
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- pandas

**Install dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt

**Setup**

Export your OpenAI API key as an environment variable:
export OPENAI_API_KEY="sk-..."

(Optional) Set a default model for all scripts:
export OPENAI_MODEL="gpt-4o"

**Usage**

Each script accepts command-line arguments for flexible input/output handling.

**Step 1: Familiarization (Inductive)**

python qualitative-analysis/Step1_Familiarization_JZQA.py \
  --input data/inputs/GroupOneTranscript_Inductive.xlsx \
  --output-dir outputs/inductive \
  --model gpt-4o

Optional flags:
	•	--sheet : specify an Excel sheet
	•	--temperature : sampling temperature (default: 1.0)
	•	--max-tokens : max tokens in response (default: 8192)
	•	--max-chars : truncate input text to this length (default: 15000)


**Step 2: Initial Coding (Inductive)**

python qualitative-analysis/Step2_Coding_JZQA.py \
  --input data/inputs/GroupOneTranscript_Inductive.xlsx \
  --output-dir outputs/inductive \
  --step1-path outputs/inductive/Step1_Familiarization.txt

data/inputs/              # user drops transcript here
outputs/inductive/        # Step1/2/3 artifacts live here

Expected output: text file with initial codes extracted from the transcript.
Writes: Step2_Coded_Responses.csv, Step2_Prompt.txt, Step2_RunMetadata.json, Step2_JSON_Failures.log

**Step 3: Codebook Development (Inductive)**

**Initial grouping only**
```bash
python qualitative-analysis/Step3_FinalCodebook_JZQA.py \
  --run-initial-grouping \
  --step2-path outputs/inductive/Step2_Coded_Responses.csv \
  --step1-path outputs/inductive/Step1_Familiarization.txt \
  --output-dir outputs/inductive \
  --model gpt-4o

Refinement only (assigns any remaining “Initial Codes” into Final Codes or creates a new Final Code when needed)
python qualitative-analysis/Step3_FinalCodebook_JZQA.py \
  --run-refinement \
  --output-dir outputs/inductive

Quote selection only (adds representative “Example” quotes for each Final Code)
python qualitative-analysis/Step3_FinalCodebook_JZQA.py \
  --run-quote-selection \
  --output-dir outputs/inductive

End‑to‑end
python qualitative-analysis/Step3_FinalCodebook_JZQA.py \
  --run-initial-grouping --run-refinement --run-quote-selection \
  --step2-path outputs/inductive/Step2_Coded_Responses.csv \
  --step1-path outputs/inductive/Step1_Familiarization.txt \
  --output-dir outputs/inductive

Common options
	•	--model (default from OPENAI_MODEL or gpt-4o)
	•	--temperature / --max-tokens (grouping)
	•	--refine-temperature / --refine-max-tokens
	•	--quote-temperature / --quote-max-tokens
	•	--min-support (default: 2) — filter low‑frequency initial codes in grouping
	•	--top-examples-per-code (default: 3)
	•	--max-rows (limit rows from Step 2 for quick tests)
	•	--max-chars (truncate grouping prompt body)
	•	--no-save-prompt (skip saving prompt text artifacts)

Step 3 outputs (in outputs/inductive/)
	•	Step3_Codebook.json — machine‑readable final codebook
	•	Step3_Codebook.txt — human‑readable codebook
	•	final_codebook.xlsx — workbook with sheets:
	•	Final Codebook (includes “Example” & “Example Source Row” after quote selection)
	•	Sampled Codes, Missing Initial Codes (refinement log), Initial Codes
	•	Prompt & run artifacts:
	•	Step3_Prompt_Grouping.txt, Step3_Prompt_Refinement.txt, Step3_Prompt_Quote.txt (unless --no-save-prompt)
	•	Step3_RunMetadata.json, Step3_InputSnapshot.csv
	•	Transcript_Text_Only.csv, initial_code_reference.csv, initial_code_reference_sampled.csv, sampled_initial_code_numbers.txt
	•	Grouped_Codes_Raw.txt, refinement_log.csv, Refinement_History/…

Expected output: structured codebook with definitions and examples.

**Deductive Coding**

python qualitative-analysis/Deductive_Coding.py \
  --input data/inputs/GroupOneTranscript_Deductive.xlsx \
  --output-dir outputs/deductive \
  --codebook outputs/inductive/Final_Codebook.txt

Expected output: dataset annotated with deductive codes from the provided codebook.

Inputs & Outputs
	•	Inputs: Excel or CSV transcripts placed in data/inputs/
	•	Outputs: Saved to outputs/inductive or outputs/deductive depending on the script
	•	Output files are plain text (or CSV) to facilitate further analysis

Example Workflow
	1.	Step 1 – Summarize and familiarize yourself with the dataset
	2.	Step 2 – Generate initial inductive codes
	3.	Step 3 – Consolidate into a codebook
	4.	Deductive Coding – Apply codebook to new transcripts

Sample Data

Anonymized sample transcripts are provided in data/samples/ for testing.
