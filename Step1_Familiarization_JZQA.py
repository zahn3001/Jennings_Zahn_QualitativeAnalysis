from openai import OpenAI
import os
import pandas as pd
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

parser = argparse.ArgumentParser(description="Step 1: Familiarization summary using Braun & Clarke framework")
parser.add_argument("--input", type=Path, default=Path("data/inputs/GroupOneTranscript_Inductive.xlsx"), help="Path to the input Excel transcript")
parser.add_argument("--sheet", type=str, default=None, help="Optional sheet name in the Excel file")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/inductive"), help="Directory to save outputs")
parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o"), help="OpenAI model name")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for the model")
parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for model response")
parser.add_argument("--max-chars", type=int, default=15000, help="Maximum characters of dataset to send to the model")
parser.add_argument("--preview-chars", type=int, default=800, help="Number of characters to preview in console")
parser.add_argument("--no-save-prompt", action="store_true", help="Do not save the full prompt to disk")
args = parser.parse_args()

if not args.input.exists():
    raise FileNotFoundError(f"Input file not found: {args.input}")

suffix = args.input.suffix.lower()
try:
    if suffix in {".xlsx", ".xls"}:
        if args.sheet:
            df = pd.read_excel(args.input, sheet_name=args.sheet)
        else:
            df = pd.read_excel(args.input)
    elif suffix in {".csv"}:
        df = pd.read_csv(args.input)
    elif suffix in {".tsv"}:
        df = pd.read_csv(args.input, sep="\t")
    else:
        # Fallback: try CSV
        df = pd.read_csv(args.input)
except ImportError as e:
    # Common case: openpyxl not installed for Excel
    raise ImportError("Reading Excel files requires 'openpyxl'. Install it via 'pip install openpyxl'.") from e

csv_text = df.to_csv(index=False)
was_truncated = len(csv_text) > args.max_chars
if was_truncated:
    file_content = csv_text[: args.max_chars] + "\n...[truncated for brevity]"
else:
    file_content = csv_text

print(f"Loaded: {args.input}")
preview = file_content[: args.preview_chars]
print(f"Dataset Preview (first {args.preview_chars} chars):\n{preview}")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable not set. See README for setup.")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the full Braun & Clarke excerpt as a separate object
braun_clarke_excerpt = """
Phase 1: Familiarization involves immersing yourself in the data through repeated reading and active engagement to identify initial patterns, meanings, and ideas. 
Avoid skipping content; this step lays the foundation for coding. Begin taking notes but avoid formal coding at this stage.
"""

# System Message defining AI's role and methodology
system_message = {
    "role": "system",
    "content": (
        "You are an AI assistant trained to perform thematic analysis based on Braun & Clarke’s 6-phase framework.\n\n"
        "**Current Focus: Step 1 – Familiarization with the Data**\n"
        "- Read the entire dataset thoroughly (repeated reading encouraged).\n"
        "- Look for recurring ideas, broad topics, and initial patterns.\n"
        "- Do not perform coding or theme categorization yet.\n"
        "- Take note of key concepts or areas of interest for future analysis.\n"
        "- This summary will serve as a foundation for more detailed analysis later."
    )
}

def save_prompt(output_dir: Path, system_msg: dict, user_msg: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "Step1_Prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("--- SYSTEM ---\n")
        f.write(system_msg.get("content", ""))
        f.write("\n\n--- USER ---\n")
        f.write(user_msg.get("content", ""))
    print(f"Saved: {prompt_path}")

def save_run_metadata(output_dir: Path, args_obj, source_path: Path, rows: int, cols: int, truncated: bool):
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": "Step1_Familiarization_JZQA.py",
        "input_path": str(source_path),
        "output_dir": str(output_dir),
        "rows": rows,
        "cols": cols,
        "model": args_obj.model,
        "temperature": args_obj.temperature,
        "max_tokens": args_obj.max_tokens,
        "max_chars": args_obj.max_chars,
        "truncated": truncated,
    }
    path = output_dir / "Step1_RunMetadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {path}")

def save_dataset_snapshot(output_dir: Path, content: str):
    path = output_dir / "Step1_Dataset_Truncated.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {path}")

def save_output(output_dir: Path, output_text: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "Step1_Familiarization.txt"
    with open(filename, "w", encoding="utf-8") as txt_file:
        txt_file.write(output_text)
    print(f"Saved: {filename}")

def process_step(model_name: str, temperature: float, max_tokens: int, output_dir: Path, save_prompt_flag: bool):
    user_message = {
        "role": "user",
        "content": (
            "Step 1: Familiarization with Data\n\n"
            "Review the dataset and provide a high-level summary of broad themes, patterns, and key ideas. Follow Braun & Clarke’s methodology for familiarization (2006) as described below:\n\n"
            f"{braun_clarke_excerpt}\n\n"
            f"**Dataset:**\n{file_content}\n\n"
            "### **Expected Output:**\n"
            "- A structured summary of key themes, patterns, and conversation topics.\n"
            "- Ensure the output aligns with the familiarization phase (repeated reading, note-taking).\n"
            "- Avoid specific codes or formal theme names at this stage."
        )
    }

    if save_prompt_flag:
        save_prompt(output_dir, system_message, user_message)

    response = client.chat.completions.create(
        model=model_name,
        messages=[system_message, user_message],
        response_format={"type": "text"},
        temperature=temperature,
        max_tokens=max_tokens,
    )

    output_text = response.choices[0].message.content
    save_output(output_dir, output_text)
    return output_text

if __name__ == "__main__":
    # Save reproducibility artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        rows, cols = df.shape
    except Exception:
        rows, cols = None, None
    save_run_metadata(args.output_dir, args, args.input, rows, cols, was_truncated)
    save_dataset_snapshot(args.output_dir, file_content)

    process_step(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        save_prompt_flag=not args.no_save_prompt,
    )
    print("Step 1 Complete")