# Quickstart Guide

## 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/qualitative-analysis.git
cd qualitative-analysis

**## 2. Set Up Python Environment**
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

**3. Configure API Key**
Export your OpenAI key:
export OPENAI_API_KEY="sk-..."

(Optional) set a default model:
export OPENAI_MODEL="gpt-4o"

**4. Run an Example**

Run Step 1 with a sample transcript:
python qualitative-analysis/Step1_Familiarization_JZQA.py \
  --input data/samples/sample_transcript.xlsx \
  --output-dir outputs/inductive

Run Step 2:
python qualitative-analysis/Step2_Coding_JZQA.py \
  --input data/samples/sample_transcript.xlsx \
  --output-dir outputs/inductive \
  --step1-path outputs/inductive/Step1_Familiarization.txt

Run Step 3:
Run grouping, refinement, and quote selection:
```bash
python qualitative-analysis/Step3_FinalCodebook_JZQA.py \
  --run-initial-grouping --run-refinement --run-quote-selection \
  --step2-path outputs/inductive/Step2_Coded_Responses.csv \
  --step1-path outputs/inductive/Step1_Familiarization.txt \
  --output-dir outputs/inductive

**5. Workflow Overview**
	1.	Step 1 – Familiarization
	2.	Step 2 – Initial Coding
	3.	Step 3 – Codebook Development
	4.	Deductive Coding – apply final codebook
