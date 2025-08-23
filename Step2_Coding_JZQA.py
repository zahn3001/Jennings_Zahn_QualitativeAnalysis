import csv
from openai import OpenAI
import os
import pandas as pd
import time
import re
import json
import argparse
from pathlib import Path
from datetime import datetime


# API key guard
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable not set. See README for setup.")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to load Step 1 Familiarization output
def load_step1_output(step1_path: Path) -> str:
    try:
        with open(step1_path, "r", encoding="utf-8") as file:
            content = file.read()
            print(f"Loaded Step 1 output: {step1_path}")
            return content
    except FileNotFoundError:
        print(f"Step 1 output not found at {step1_path}. Proceeding without it.")
        return "Step 1 output not found. Proceeding without it."


# CLI argument parsing
parser = argparse.ArgumentParser(description="Step 2: Initial Coding (Braun & Clarke)")
parser.add_argument("--input", type=Path, default=Path("data/inputs/GroupOneTranscript_Inductive.xlsx"), help="Path to the input transcript (.xlsx/.xls/.csv/.tsv)")
parser.add_argument("--sheet", type=str, default=None, help="Optional Excel sheet name")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/inductive"), help="Directory to save outputs")
parser.add_argument("--step1-path", type=Path, default=Path("outputs/inductive/Step1_Familiarization.txt"), help="Path to Step 1 familiarization output")
parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o"), help="OpenAI model name")
parser.add_argument("--temperature", type=float, default=1.1, help="Model temperature")
parser.add_argument("--max-tokens", type=int, default=750, help="Max tokens per code generation call")
parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to process (default: all)")
parser.add_argument("--no-save-prompt", action="store_true", help="Do not save prompt artifacts")
parser.add_argument("--temp-fallback", type=float, default=None, help="Temperature for fallback retry (default: base-0.2)")
parser.add_argument("--temp-strict", type=float, default=None, help="Temperature for strict retry (default: 0.5)")
parser.add_argument("--temp-ultra", type=float, default=None, help="Temperature for ultra-final retry (default: 0.1)")
parser.add_argument("--tokens-strict", type=int, default=None, help="Max tokens for strict retry (default: min(500, --max-tokens))")
parser.add_argument("--tokens-ultra", type=int, default=None, help="Max tokens for ultra-final retry (default: min(300, --max-tokens))")
args = parser.parse_args()


args.output_dir.mkdir(parents=True, exist_ok=True)

# Effective per-attempt settings (can be overridden via CLI)
temp_fallback = args.temp_fallback if args.temp_fallback is not None else max(0.1, round(args.temperature - 0.2, 2))
temp_strict = args.temp_strict if args.temp_strict is not None else 0.5
temp_ultra = args.temp_ultra if args.temp_ultra is not None else 0.1
tokens_strict = args.tokens_strict if args.tokens_strict is not None else min(500, args.max_tokens)
tokens_ultra = args.tokens_ultra if args.tokens_ultra is not None else min(300, args.max_tokens)


# System Message defining AI's role and methodology
system_message = {
    "role": "system",
    "content": (
        "You are an AI assistant and expert in medical education, trained to conduct thematic analysis using Braun & Clarke’s 6-step methodology.\n\n"
        "**Current Focus: Step 2 – Generating Initial Codes**\n"
        "- Identify and label meaningful features in each student response.\n"
        "- A 'code' is a brief phrase capturing a significant idea related to the research aim.\n"
        "- Focus on content, contradictions, and nuances—do not reduce to themes.\n"
        "- Treat each data segment with equal importance.\n"
        "- Work iteratively and avoid assumptions or biases.\n"
        "- Use clear, concise language to generate consistent, comparable codes."
    )
}

# File-type-aware transcript loader
if not args.input.exists():
    raise FileNotFoundError(f"Input file not found: {args.input}")

suffix = args.input.suffix.lower()
if suffix in {".xlsx", ".xls"}:
    df = pd.read_excel(args.input, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.input)
elif suffix == ".csv":
    df = pd.read_csv(args.input)
elif suffix == ".tsv":
    df = pd.read_csv(args.input, sep="\t")
else:
    # Fallback: try CSV
    df = pd.read_csv(args.input)
print(f"Loaded transcript: {args.input}")

# Load Step 1 familiarization summary
step1_output = load_step1_output(args.step1_path)

# Prepare common output paths
coded_csv_path = args.output_dir / "Step2_Coded_Responses.csv"
json_fail_log = args.output_dir / "Step2_JSON_Failures.log"
prompt_txt_path = args.output_dir / "Step2_Prompt.txt"
run_meta_path = args.output_dir / "Step2_RunMetadata.json"

# Helpers for reproducibility artifacts
def save_prompt(path: Path, system_msg: dict, user_msg: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write("--- SYSTEM ---\n" + system_msg.get("content", "") + "\n\n")
        f.write("--- USER TEMPLATE (one example) ---\n" + user_msg.get("content", ""))
    print(f"Saved: {path}")

def save_run_metadata(path: Path, rows: int, cols: int):
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": "Step2_Coding_JZQA.py",
        "input_path": str(args.input),
        "output_dir": str(args.output_dir),
        "step1_path": str(args.step1_path),
        "rows": rows,
        "cols": cols,
        "model": args.model,
        "temperature_base": args.temperature,
        "temperature_fallback": temp_fallback,
        "temperature_strict": temp_strict,
        "temperature_ultra": temp_ultra,
        "max_tokens_base": args.max_tokens,
        "max_tokens_strict": tokens_strict,
        "max_tokens_ultra": tokens_ultra,
        "max_rows": args.max_rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {path}")

# Create an example user_message using the first Student response (if available) for prompt artifact
example_student_text = None
for _, r in df.iterrows():
    if "Student" in str(r.get("Speaker", "")):
        example_student_text = r.get("Text", "")
        break

example_user_message = {
    "role": "user",
    "content": (
        "Step 2: Generating Initial Codes\n\n"
        "Generate 1–3 concise codes that capture the key ideas or sentiments in the following student response. "
        "Think iteratively—consider meaning, nuance, emotions, sentiments, and contradictions before deciding on codes. "
        "Refer to Braun & Clarke’s guidance for initial coding and use the familiarization summary for context.\n\n"
        "### Step 1 Familiarization Summary:\n"
        f"{step1_output}\n\n"
        "### Student Response:\n"
        f"\"{(example_student_text or 'N/A')}\"\n\n"
        "### Output Instructions:\n"
        "- Respond using strict JSON format only.\n"
        "- Return a list of dictionaries (maximum of 3).\n"
        "- Each dictionary must contain:\n"
        "  - \"code\": <short label>\n"
        "  - \"definition\": <brief description>\n"
        "  - \"reasoning\": <why it fits the student response>\n"
    )
}

if not args.no_save_prompt:
    save_prompt(prompt_txt_path, system_message, example_user_message)

try:
    r, c = df.shape
except Exception:
    r, c = None, None
save_run_metadata(run_meta_path, r, c)

for i in range(1, 4):  # Code 1..3 are generated
    for suffix in ["", " Definition", " Reasoning"]:
        col = f"Code {i}{suffix}"
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(object)
            df[col] = ""



print(f"Prompting AI for coding... Outputs will be saved to: {coded_csv_path}")

iteration = 1
row_iterator = df.head(args.max_rows).iterrows() if args.max_rows else df.iterrows()
for index, row in row_iterator:
    print(f"Coding line {index + 1}/{len(df)}: {row['Speaker']}")
    if "Student" in str(row["Speaker"]):  # Only process Student responses
        student_text = row["Text"]

        # AI prompt
        user_message = {
            "role": "user",
            "content": (
                "Step 2: Generating Initial Codes\n\n"
                "Generate 1–3 concise codes that capture the key ideas or sentiments in the following student response. "
                "Think iteratively—consider meaning, nuance, emotions, sentiments, and contradictions before deciding on codes. "
                "Refer to Braun & Clarke’s guidance for initial coding and use the familiarization summary for context.\n\n"
                "### Step 1 Familiarization Summary:\n"
                "Use the following summary to understand the broader context of the data and help you infer meaning from responses that may seem ambiguous without additional context:\n\n"
                f"{step1_output}\n\n"
                "### Student Response:\n"
                f"\"{student_text}\"\n\n"
                "### Output Instructions:\n"
                "- Respond using strict JSON format only.\n"
                "- Return a list of dictionaries (maximum of 3).\n"
                "- Each dictionary must contain:\n"
                "  - \"code\": <short label>\n"
                "  - \"definition\": <brief description>\n"
                "  - \"reasoning\": <why it fits the student response>\n\n"
                "### Example:\n"
                "[\n"
                "  {\n"
                "    \"code\": \"Feedback Loop Awareness\",\n"
                "    \"definition\": \"Recognizing how feedback from the AI influences learning\",\n"
                "    \"reasoning\": \"The student describes how immediate feedback changed their understanding.\"\n"
                "  },\n"
                "  {\n"
                "    \"code\": \"Confidence Building\",\n"
                "    \"definition\": \"Gaining assurance through AI-based practice\",\n"
                "    \"reasoning\": \"The student mentions that using the tool helped them feel more prepared.\"\n"
                "  }\n"
                "]"
            )
        }

        # Send request to OpenAI
        response_ai = client.chat.completions.create(
            model=args.model,
            messages=[system_message, user_message],
            response_format={"type": "text"},
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )

        # Extract and clean AI JSON response
        ai_generated_codes = response_ai.choices[0].message.content.strip()
        ai_generated_codes = re.sub(r"^```(?:json)?|```$", "", ai_generated_codes, flags=re.IGNORECASE).strip()

        try:
            code_blocks = json.loads(ai_generated_codes)
        except json.JSONDecodeError:
            print(f"⚠️ JSON parsing error at row {index + 1}. Saving raw content.")
            code_blocks = []

        # Check if any code blocks were parsed
        if not code_blocks:
            print(f"Re-prompting line {index + 1} — no valid codes detected.")
            fallback_prompt = {
                "role": "user",
                "content": (
                    "Step 2: Generating Initial Codes\n\n"
                    "Generate 1–3 concise codes that capture the key ideas in the following student response. "
                    "Think iteratively—consider meaning, nuance, and contradictions before deciding on codes. "
                    "Refer to Braun & Clarke’s guidance for initial coding and use the familiarization summary for context.\n\n"
                    "### Step 1 Familiarization Summary:\n"
                    f"{step1_output}\n\n"
                    "Your last response did not include any valid codes in JSON format. Please try again.\n\n"
                    "Use this format exactly for each code (up to 3):\n"
                    "[\n"
                    "  {\n"
                    "    \"code\": \"<short label>\",\n"
                    "    \"definition\": \"<short definition>\",\n"
                    "    \"reasoning\": \"<why it fits the student response>\"\n"
                    "  }\n"
                    "]\n\n"
                    f"### Student Response:\n\"{student_text}\""
                )
            }

            response_ai = client.chat.completions.create(
                model=args.model,
                messages=[system_message, fallback_prompt],
                response_format={"type": "text"},
                temperature=temp_fallback,
                max_tokens=args.max_tokens
            )

            ai_generated_codes = response_ai.choices[0].message.content.strip()
            ai_generated_codes = re.sub(r"^```(?:json)?|```$", "", ai_generated_codes, flags=re.IGNORECASE).strip()
            try:
                code_blocks = json.loads(ai_generated_codes)
            except json.JSONDecodeError:
                if not code_blocks:
                    print(f"⚠️ JSON parsing error at row {index + 1} on fallback. Saving raw content.")
                    with open(json_fail_log, "a", encoding="utf-8") as f:
                        f.write(f"\n\nLine {index + 1} fallback response:\n{ai_generated_codes}")

                    print(f"Final re-prompting line {index + 1} with strict JSON enforcement.")
                    strict_prompt = {
                        "role": "user",
                        "content": (
                            "Step 2: Generating Initial Codes\n\n"
                            "Generate 1–3 concise codes that capture the key ideas in the following student response. "
                            "Think iteratively—consider meaning, nuance, and contradictions before deciding on codes. "
                            "Refer to Braun & Clarke’s guidance for initial coding and use the familiarization summary for context.\n\n"
                            "### Step 1 Familiarization Summary:\n"
                            f"{step1_output}\n\n"
                            "⚠️ Your last response was not valid JSON. This is your final chance to return correctly formatted output.\n\n"
                            "⚠️ VERY IMPORTANT: Only return valid JSON. No explanations. No markdown formatting.\n"
                            "Return a list (max 3 items), where each item is a dictionary with all 3 keys:\n"
                            "- code: <short phrase>\n"
                            "- definition: <brief definition>\n"
                            "- reasoning: <why the code fits>\n\n"
                            "Return only JSON. Do not use triple backticks or include any extra text.\n\n"
                            f"### Student Response:\n\"{student_text}\""
                        )
                    }

                    final_response = client.chat.completions.create(
                        model=args.model,
                        messages=[system_message, strict_prompt],
                        response_format={"type": "text"},
                        temperature=temp_strict,
                        max_tokens=tokens_strict
                    )

                    final_text = final_response.choices[0].message.content.strip()
                    final_text = re.sub(r"^```(?:json)?|```$", "", final_text, flags=re.IGNORECASE).strip()
                    try:
                        code_blocks = json.loads(final_text)
                    except json.JSONDecodeError:
                        print(f"⚠️ JSON parsing failed on third attempt at row {index + 1}.")
                        with open(json_fail_log, "a", encoding="utf-8") as f:
                            f.write(f"\n\nLine {index + 1} third attempt:\n{final_text}")
                        code_blocks = []

                    # Ultra-conservative final retry if still no code_blocks
                    if not code_blocks:
                        print(f"⚠️ JSON still invalid. Final low-temp retry at row {index + 1}.")
                        ultra_final_response = client.chat.completions.create(
                            model=args.model,
                            messages=[system_message, strict_prompt],
                            response_format={"type": "text"},
                            temperature=temp_ultra,
                            max_tokens=tokens_ultra
                        )
                        ultra_final_text = ultra_final_response.choices[0].message.content.strip()
                        ultra_final_text = re.sub(r"^```(?:json)?|```$", "", ultra_final_text, flags=re.IGNORECASE).strip()
                        try:
                            code_blocks = json.loads(ultra_final_text)
                        except json.JSONDecodeError:
                            print(f"❌ JSON parsing failed on fourth attempt at row {index + 1}. Skipping line.")
                            with open(json_fail_log, "a", encoding="utf-8") as f:
                                f.write(f"\n\nLine {index + 1} fourth attempt:\n{ultra_final_text}")
                            code_blocks = []

        # Assign values to columns
        for i in range(3):
            code_col = f"Code {i+1}"
            def_col = f"Code {i+1} Definition"
            reason_col = f"Code {i+1} Reasoning"

            if code_col not in df.columns:
                df[code_col] = ""
                df[def_col] = ""
                df[reason_col] = ""

            if i < len(code_blocks):
                block = code_blocks[i]
                df.at[index, code_col] = block.get("code", "")
                df.at[index, def_col] = block.get("definition", "")
                df.at[index, reason_col] = block.get("reasoning", "")

        # Save progress after each student response
        df.to_csv(coded_csv_path, index=False)

print("Validating missing definitions or explanations...")

row_iterator = df.head(args.max_rows).iterrows() if args.max_rows else df.iterrows()
for index, row in row_iterator:
    if "Student" in str(row["Speaker"]):
        student_text = row["Text"]
        for i in range(3):
            code = row.get(f"Code {i+1}", "")
            def_col = f"Code {i+1} Definition"
            reason_col = f"Code {i+1} Reasoning"
            definition = row.get(def_col, "")
            reasoning = row.get(reason_col, "")

            missing_definition = (
                not definition.strip()
                or len(definition.strip()) < 10
                or "definition" in definition.lower()
            )
            missing_reasoning = (
                not reasoning.strip()
                or len(reasoning.strip()) < 10
                or "reasoning" in reasoning.lower()
            )

            if code and (missing_definition or missing_reasoning):
                print(f"Fixing missing info for line {index + 1}, Code {i+1}: {code}")
                prompt_parts = [f"The code is: {code}"]
                if missing_definition:
                    prompt_parts.append("Please provide a brief definition for this code.")
                if missing_reasoning:
                    prompt_parts.append("Please provide a brief explanation of why this code fits the student response.")

                followup_prompt = {
                    "role": "user",
                    "content": (
                        f"A previous response was missing some information.\n\n"
                        f"### Student Response:\n\"{student_text}\"\n\n"
                        + "\n".join(prompt_parts)
                    )
                }

                followup_response = client.chat.completions.create(
                    model=args.model,
                    messages=[system_message, followup_prompt],
                    response_format={"type": "text"},
                    temperature=args.temperature,
                    max_tokens=150
                )

                followup_text = followup_response.choices[0].message.content.strip().splitlines()
                for line in followup_text:
                    if missing_definition and line.lower().startswith("definition:"):
                        df.at[index, def_col] = line.split(":", 1)[1].strip()
                        missing_definition = False
                    elif missing_reasoning and line.lower().startswith("reasoning:"):
                        df.at[index, reason_col] = line.split(":", 1)[1].strip()
                        missing_reasoning = False

        # Save after each student response validation
        df.to_csv(coded_csv_path, index=False)

print("Coding complete.\n")

# Drop deprecated aggregate columns if present, then save final CSV
for col in ["Codes", "Code Definitions", "Code Reasoning"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
df.to_csv(coded_csv_path, index=False)
print(f"Step 2 complete: Coded responses saved to {coded_csv_path}")
