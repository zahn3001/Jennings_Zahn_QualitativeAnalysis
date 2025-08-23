import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from openai import OpenAI

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="Deductive Coding with Codebook (applies existing codes to transcript)")

# Inputs / outputs
parser.add_argument("--input", type=Path, default=Path("data/inputs/GroupOneTranscript_Deductive.xlsx"),
                    help="Path to the transcript file (.xlsx/.xls/.csv/.tsv) with first 3 rows as codebook and row 4 as headers.")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/deductive"),
                    help="Directory to save outputs and artifacts.")

# Model settings
parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o"), help="OpenAI model name")
parser.add_argument("--temperature", type=float, default=0.4, help="Base temperature for coding prompts")
parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens per prompt")

# Run controls
parser.add_argument("--context-window", type=int, default=1, help="Rows of surrounding context on each side")
parser.add_argument("--start-row", type=int, default=0, help="Start index within student rows (0-based)")
parser.add_argument("--end-row", type=int, default=None, help="End index within student rows (exclusive); default: all")
parser.add_argument("--rate-limit-sleep", type=float, default=2.0, help="Seconds to sleep between rows")
parser.add_argument("--no-save-prompt", action="store_true", help="Do not save the prompt template artifact")

# Code selection controls
parser.add_argument("--code-mode", type=str, choices=["all", "last", "custom", "first_n", "every_nth"], default="all",
                    help="Which codes to evaluate")
parser.add_argument("--custom-codes", nargs="*", default=[], help="List of specific code names to evaluate when --code-mode=custom")
parser.add_argument("--n", type=int, default=5, help="N for first_n/every_nth modes")

# Advanced input parsing (for non-standard files)
parser.add_argument("--header-rows", type=int, default=4, help="Number of header rows before data (default 4 as in template)")
parser.add_argument("--code-start-col", type=int, default=3, help="Column index where code columns begin (0-based; default 3)")

args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# API client (guard first)
# ----------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable not set. See README for setup.")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Paths for artifacts
# ----------------------------
coded_output_path = args.output_dir / "AI_Coded_Transcript_with_Reasoning.csv"
backup_output_path = args.output_dir / "AI_Coded_Transcript_with_Reasoning_backup.csv"
prompt_template_path = args.output_dir / "Deductive_Prompt_Template.txt"
run_metadata_path = args.output_dir / "Deductive_RunMetadata.json"
codebook_snapshot_path = args.output_dir / "Deductive_CodebookSnapshot.json"
input_snapshot_path = args.output_dir / "Deductive_InputSnapshot.csv"

# ----------------------------
# Load transcript (supports xlsx/csv/tsv)
# ----------------------------
if not args.input.exists():
    raise FileNotFoundError(f"Input file not found: {args.input}")

suffix = args.input.suffix.lower()
if suffix in {".xlsx", ".xls"}:
    # Read without header so we can set headers from the 4th row (index 3)
    df = pd.read_excel(args.input, header=None)
elif suffix == ".csv":
    df = pd.read_csv(args.input, header=None)
elif suffix == ".tsv":
    df = pd.read_csv(args.input, header=None, sep="\t")
else:
    # Fallback: try CSV
    df = pd.read_csv(args.input, header=None)

# Set headers using the (header_rows-1)th row, then keep original for codebook extraction
header_row_idx = args.header_rows - 1
if len(df) <= header_row_idx:
    raise ValueError(f"File appears too short to contain the expected header rows ({args.header_rows}).")

df.columns = df.iloc[header_row_idx]

# Save a snapshot of the input we actually used
try:
    df.to_csv(input_snapshot_path, index=False)
    print(f"Saved: {input_snapshot_path}")
except Exception:
    pass

# ----------------------------
# Extract codebook (first three rows) and determine target codes
# ----------------------------
code_start = args.code_start_col

code_names = pd.Index(df.iloc[0, code_start:])
code_names = code_names[code_names.notna() & (code_names != "")]

# Descriptions and examples are from rows 1 and 2
raw_desc = list(df.iloc[1, code_start:])
raw_ex = list(df.iloc[2, code_start:])

# Zip into snapshot
codebook_items = []
for i, name in enumerate(code_names):
    desc = raw_desc[i] if i < len(raw_desc) else ""
    ex = raw_ex[i] if i < len(raw_ex) else ""
    codebook_items.append({"code": str(name), "definition": str(desc), "example": str(ex)})

with open(codebook_snapshot_path, "w", encoding="utf-8") as f:
    json.dump(codebook_items, f, indent=2, ensure_ascii=False)
print(f"Saved: {codebook_snapshot_path}")

# Resolve target codes based on mode
if args.code_mode == "all":
    TARGET_CODES = code_names.tolist()
elif args.code_mode == "last":
    TARGET_CODES = [code_names[-1]] if len(code_names) else []
elif args.code_mode == "custom":
    TARGET_CODES = [code for code in code_names if code in set(args.custom_codes)]
elif args.code_mode == "first_n":
    TARGET_CODES = code_names[: args.n].tolist()
elif args.code_mode == "every_nth":
    TARGET_CODES = code_names[:: args.n].tolist()
else:
    raise ValueError("Invalid --code-mode selected")

print("Codes selected for evaluation:")
print(TARGET_CODES)

# ----------------------------
# Build working DataFrames
# ----------------------------
# Drop the header rows from the full transcript (keep all speakers for context)
base_df = df.iloc[args.header_rows:].copy()
# Drop unnamed filler columns
base_df = base_df.loc[:, ~base_df.columns.astype(str).str.contains(r"^Unnamed")] 

# Create student-only view for coding
if "Speaker" not in base_df.columns or "Text" not in base_df.columns:
    raise KeyError("Expected columns 'Speaker' and 'Text' not found after header parsing.")

df_students = base_df[base_df["Speaker"].astype(str).str.contains("Student", na=False)].copy()
print(f"Number of student responses selected for coding: {len(df_students)}")

# Reasoning columns for each code
for code in code_names:
    col = f"Reasoning - {code}"
    if col not in df_students.columns:
        df_students[col] = ""

metadata_cols = [c for c in ["Timestamp", "Speaker", "Text"] if c in df_students.columns]
reasoning_cols = [f"Reasoning - {code}" for code in code_names]

# ----------------------------
# System message & prompt template
# ----------------------------
system_message = {
    "role": "system",
    "content": (
        "You are a qualitative coding assistant applying a predefined codebook. "
        "You will carefully evaluate whether ONE code applies to a single TARGET student's response. "
        "Use the code's definition and example to guide your decision. "
        "If it clearly does NOT apply, reply exactly '0'. If it applies, justify in 1–2 sentences why. "
        "Never code based solely on surrounding context; context is provided only to interpret the TARGET response."
    ),
}

PROMPT_TEMPLATE = (
    "You are a qualitative coding assistant. Given a single target student's response and some surrounding context, "
    "determine whether the following code applies to the student's response.\n\n"
    "### CODE:\n{code}\n"
    "Definition: {definition}\n"
    "Example: {example}\n\n"
    "### TARGET STUDENT RESPONSE (Target for Coding):\n\"{target}\"\n\n"
    "### CONTEXT (Do NOT code based on this; use only to interpret the above):\n{context}\n\n"
    "### INSTRUCTIONS:\n"
    "Consider whether the code meaningfully applies to the **target student's response**, using the definition and example as guidance. "
    "Read the entire target student's response carefully and consider nuance, emotions, and implied meaning relevant to the code. "
    "Apply the code when there is clear relevance or strong implication in the target student's response. Do not code based on other speakers. "
    "If the code applies, explain in 1–2 sentences why it fits. If it clearly does not apply, reply exactly: '0'."
)

# Save prompt template artifact
if not args.no_save_prompt:
    try:
        example_target = (df_students.iloc[0]["Text"] if not df_students.empty else "")
    except Exception:
        example_target = ""
    example_code = TARGET_CODES[0] if TARGET_CODES else ""
    try:
        idx0 = list(code_names).index(example_code) if example_code else 0
    except ValueError:
        idx0 = 0
    example_def = str(raw_desc[idx0]) if idx0 < len(raw_desc) else ""
    example_ex = str(raw_ex[idx0]) if idx0 < len(raw_ex) else ""
    example_prompt = PROMPT_TEMPLATE.format(
        code=example_code,
        definition=example_def,
        example=example_ex,
        target=example_target,
        context="[context window omitted in template]",
    )
    with open(prompt_template_path, "w", encoding="utf-8") as f:
        f.write("--- SYSTEM ---\n" + system_message["content"] + "\n\n")
        f.write("--- USER TEMPLATE ---\n" + example_prompt)
    print(f"Saved: {prompt_template_path}\n")

# ----------------------------
# Run metadata
# ----------------------------
run_meta = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "script": "Deductive_JZQA.py",
    "input_path": str(args.input),
    "output_dir": str(args.output_dir),
    "model": args.model,
    "temperature": args.temperature,
    "max_tokens": args.max_tokens,
    "context_window": args.context_window,
    "start_row": args.start_row,
    "end_row": args.end_row,
    "rate_limit_sleep": args.rate_limit_sleep,
    "code_mode": args.code_mode,
    "custom_codes": args.custom_codes,
    "n": args.n,
    "header_rows": args.header_rows,
    "code_start_col": args.code_start_col,
    "num_student_rows": int(len(df_students)),
    "num_total_codes": int(len(code_names)),
    "num_target_codes": int(len(TARGET_CODES)),
}
with open(run_metadata_path, "w", encoding="utf-8") as f:
    json.dump(run_meta, f, indent=2)
print(f"Saved: {run_metadata_path}")

# ----------------------------
# Coding helpers
# ----------------------------

def get_context(idx_in_student_df: int, context_window: int) -> str:
    """Return a string of neighbor rows around the given student row index, from the full base_df (all speakers)."""
    # Find the global index (within base_df) corresponding to this student row
    global_index = df_students.index[idx_in_student_df]
    start = max(global_index - context_window, base_df.index.min())
    end = min(global_index + context_window, base_df.index.max())
    context_rows = base_df.loc[start:end]

    chunks = []
    for idx, r in context_rows.iterrows():
        prefix = ">>>" if idx == global_index else ""
        speaker = str(r.get("Speaker", ""))
        text = str(r.get("Text", ""))
        chunks.append(f"{prefix}{speaker} {idx}: \"{text}\"")
    return "\n".join(chunks)


# ----------------------------
# Main loop
# ----------------------------
print("Prompting AI for coding...")

# Determine the slice of student rows to process
end_row = args.end_row if args.end_row is not None else len(df_students)
df_students_subset = df_students.iloc[args.start_row:end_row]

# Ensure we start with a clean output CSV (overwrite each run)
if coded_output_path.exists():
    coded_output_path.unlink()

for local_idx, (global_idx, row) in enumerate(df_students_subset.iterrows()):
    # Local index within subset (0..len-1)
    print(f"Processing row {args.start_row + local_idx + 1} of {len(df_students_subset)}")

    target_response = str(row.get("Text", ""))
    surrounding_context = get_context(df_students.index.get_loc(global_idx), args.context_window)
    surrounding_context = surrounding_context.replace(f">>>{row.get('Speaker', '')} {global_idx}: \"{target_response}\"", "").strip()

    # Clear reasoning fields for this row before re-running
    for code in TARGET_CODES:
        df_students.at[global_idx, f"Reasoning - {code}"] = ""

    for code in TARGET_CODES:
        try:
            idx = list(code_names).index(code)
        except ValueError:
            # Skip if code not found for some reason
            continue
        desc = str(raw_desc[idx]) if idx < len(raw_desc) else ""
        ex = str(raw_ex[idx]) if idx < len(raw_ex) else ""

        user_message = {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                code=code,
                definition=desc,
                example=ex,
                target=target_response,
                context=surrounding_context,
            ),
        }

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[system_message, user_message],
                response_format={"type": "text"},
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            result = (response.choices[0].message.content or "").strip()
            # Store reasoning only if not the explicit '0'
            if result.lower() != "0":
                df_students.at[global_idx, f"Reasoning - {code}"] = result
            else:
                df_students.at[global_idx, f"Reasoning - {code}"] = ""
        except Exception as e:
            print(f"Error at global row {global_idx}, code '{code}': {e}")

    # Save rolling backup after each row
    try:
        df_students.to_csv(backup_output_path, index=False)
    except Exception:
        pass

    # Append current row's results to output CSV
    row_temp_df = df_students.loc[[global_idx], ["Text"] + reasoning_cols].copy()
    # Rename columns: Reasoning - X -> X
    rename_map = {f"Reasoning - {code}": str(code) for code in code_names}
    row_temp_df.rename(columns=rename_map, inplace=True)
    row_temp_df.to_csv(coded_output_path, mode='a', header=not coded_output_path.exists(), index=False)

    # Rate limit
    time.sleep(args.rate_limit_sleep)

# Final full snapshot (optional)
try:
    temp_df = df_students[["Text"] + reasoning_cols].copy()
    temp_df.rename(columns={f"Reasoning - {c}": str(c) for c in code_names}, inplace=True)
    temp_df.to_csv(args.output_dir / "AI_Coded_Transcript_with_Reasoning_full.csv", index=False)
except Exception:
    pass

print("Coding complete.")
