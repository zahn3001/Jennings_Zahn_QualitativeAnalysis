import pandas as pd
import csv
import json
import os
import re
from openai import OpenAI

# --- Utility: Normalize grouped codes string ---
def normalize_grouped_codes(s: str) -> str:
    """
    Extract digits from any formatting like [1], (2), ' 3 ' and return a
    sorted, de-duplicated, comma-separated string like '1, 2, 3'.
    """
    nums = re.findall(r"\d+", str(s))
    if not nums:
        return ""
    unique_sorted = sorted({int(n) for n in nums})
    return ", ".join(str(n) for n in unique_sorted)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to load Step 1 Familiarization output
def load_step1_output(file_path="/Users/andrewzahn/Documents/Github Repositories/Qualitative Analysis/qualitative-analysis/New New Inductive Results/Step1_Familiarization.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "Step 1 output not found. Proceeding without it."

print("Step 1 output loaded")
print("Loading familiarization summary and dataset...")


# Load Step 1 familiarization summary
step1_output = load_step1_output()

# === Step 1 context preparation helper ===
def _prepare_step1_context(raw_text: str, max_chars: int = 2000) -> str:
    """
    Prepare a compact Step 1 familiarization context for prompts.
    Collapses whitespace and trims to max_chars to control token usage.
    """
    if not isinstance(raw_text, str):
        return ""
    compact = re.sub(r"\s+", " ", raw_text).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "…"
    return compact

step1_context = _prepare_step1_context(step1_output, max_chars=2000)
print(f"Including {len(step1_context)} chars of Step 1 context in prompts.")


# Load the dataset
file_path = "/Users/andrewzahn/Documents/Github Repositories/Qualitative Analysis/qualitative-analysis/New New Inductive Results/Step2_Coded_Responses.csv"
df = pd.read_csv(file_path)

# === Part 1: Transcript Text with Enumeration ===
transcript_df = df[["Text"]].dropna().drop_duplicates().reset_index(drop=True)
transcript_df.insert(0, "ID", range(1, len(transcript_df) + 1))
transcript_output_path = "New New Inductive Results/Transcript_Text_Only.csv"
transcript_df.to_csv(transcript_output_path, index=False)
print(f"Transcript-only file saved to {transcript_output_path}")

# === Part 2: Initial Codes and Definitions ===
# Extract all code columns by pattern: Code 1, Code 2, etc. (not Code X Reasoning/Score/Definition)
code_cols = [col for col in df.columns if re.match(r"Code \d+$", col)]
definition_cols = [col for col in df.columns if "Definition" in col]
print(f"Code columns detected: {code_cols}")
print(f"Definition columns detected: {definition_cols}")

# Check for mismatch between code and definition columns
if len(code_cols) != len(definition_cols):
    raise ValueError("Mismatch between code and definition columns.")

# Flatten and combine into list of tuples
initial_code_set = set()
for code_col, def_col in zip(code_cols, definition_cols):
    for code, definition in zip(df[code_col], df[def_col]):
        code_key = str(code).strip().lower()
        def_val = str(definition).strip()
        if pd.notna(code) and code_key not in ["", "nan", "null"]:
            initial_code_set.add((code_key, def_val))

# Convert to sorted DataFrame with enumeration
initial_codes_list = sorted(list(initial_code_set))
initial_codes_df = pd.DataFrame(initial_codes_list, columns=["Initial Code", "Definition"])
initial_codes_df.insert(0, "Number", range(1, len(initial_codes_df) + 1))

initial_codes_output_path = "New New Inductive Results/initial_code_reference.csv"
initial_codes_df.to_csv(initial_codes_output_path, index=False)
print(f"Initial codes extracted and saved to {initial_codes_output_path}")

# === Part 3: Sample 50 Initial Codes for Prompting ===
if len(initial_codes_df) < 50:
    raise ValueError("Not enough initial codes available to sample 50.")
sample_n = 50
sampled_codes_df = initial_codes_df.sample(n=sample_n, random_state=42).sort_values("Number").reset_index(drop=True)
sampled_codes_output_path = "New New Inductive Results/initial_code_reference_sampled.csv"
sampled_codes_df.to_csv(sampled_codes_output_path, index=False, columns=initial_codes_df.columns)
print(f"Sampled {sample_n} initial codes saved to {sampled_codes_output_path}")

# Save sampled numbers to a text file
sampled_numbers_path = "New New Inductive Results/sampled_initial_code_numbers.txt"
with open(sampled_numbers_path, "w") as f:
    f.write(", ".join(map(str, sampled_codes_df["Number"].tolist())))
print(f"Sampled initial code numbers saved to {sampled_numbers_path}")




# Remove the "Text" column since transcript text is no longer needed
if "Text" in df.columns:
    df = df.drop(columns=["Text"])

# Also remove "Timestamp" and "Speaker" columns if present
for col in ["Timestamp", "Speaker"]:
    if col in df.columns:
        df = df.drop(columns=[col])


print("Step 2 formatted output loaded")
print("Refined codes extracted and deduplicated.")



# For all-code mapping and missing code checks, use all initial codes
code_entries = [
    {"number": int(row["Number"]), "code": row["Initial Code"], "definition": row["Definition"]}
    for _, row in initial_codes_df.iterrows()
]

# For prompt: only use the sample
code_text_blocks = []
for _, row in sampled_codes_df.iterrows():
    code_block = (
        f"{row['Number']}. Code: {row['Initial Code']}\n"
        f"Definition: {row['Definition']}\n"
    )
    code_text_blocks.append(code_block)
full_code_text = "\n\n".join(code_text_blocks)

def run_code_grouping():
    # ===== Prompt #1: Final Code Grouping and Definition Refinement =====
    system_message = {
        "role": "system",
        "content": (
            "You are an expert qualitative researcher and expert in medical education trained to conduct rigorous qualitative analysis following Braun & Clarke’s (2006) thematic analysis guidance tasked with creating **final codes** from a set of initial codes and their definitions."
            "You have completed the first two steps of thematic analysis: familiarization with the data and generating initial codes."
            "### High-Level Context from Step 1 Familiarization (summarization of transcript)\n"
            f"{step1_context}\n\n"
            "A **final code** is a short, precise label that captures a specific, observable idea or concept in the data. It is concrete, descriptive, and based on what is explicitly stated in the responses."
            "Final codes are **NOT** themes. A theme is a broader, interpretive concept that groups multiple codes together to tell a larger story; final codes remain more specific and granular."
            "Your role is to:"
            "- **Group** related initial codes into the most fitting final code(s)"
            "- **Create** a new final code only if the initial code does not fit any existing final code"
            "- Ensure each final code has a clear, concise **definition** that captures its meaning without expanding into theem-level interpretation."
            "- Follow all requested output formats exactly."
            "Always avoid producing themes or broad categories - focus only on generating the most accurate set of **final codes** and their definitions."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            """You are reviewing 50 initial codes and definitions from a qualitative analysis of a focus group transcript. You have already seen a high‑level summary of the transcript.

            Your task is to synthesize the **most obvious** groupings into final codes. A final code is specific, concrete, and descriptive (not a broad theme). Only group codes that clearly share the same idea. If a code does not clearly fit, leave it ungrouped for later refinement.Take into consideration implicit meaning in both initial and final code names and definitions when making your decision.

            **Instructions**
            - Review **all 50** initial codes before grouping; weigh them equally.
            - Create final codes only for **clear, high-confidence groupings** based on explicit AND implicit shared meaning between initial code names or definitions.
            - You may leave any number of initial codes ungrouped at this stage.
            - Do **not** invent or assume content beyond the provided code names/definitions.
            - Use your own concise wording for each final code’s name and definition.
            - Each initial code number may appear in **at most one** final code (no duplicates across groups).
            - **Do not include** any initial code numbers that are not in the provided list.
            - Keep the focus on specific, observable ideas (final codes), **not** broader interpretive themes.

            **Output Format (EXACTLY)**
            For each final code you decide to produce, output a block in this exact format:

            **Code: [Final Code Name]**  
            **Definition:** [Concise definition grounded only in the data]  
            Grouped Codes: 1, 4, 7  
            Explanation: [1–2 sentences explaining the shared idea and why these codes belong together]

            Formatting rules:
            - `Grouped Codes:` must be plain comma‑separated integers with **no brackets or extra text** (e.g., `1, 4, 7`).
            - Do **not** include any other headers, lists, or commentary outside the blocks.
            - It is acceptable to output only a few blocks if only a few groupings are clearly obvious.

            **Quality checks (apply before returning your answer)**
            - No bracketed numbers (e.g., `[1]`) and no duplicates across any groups.
            - Every number under `Grouped Codes:` must be from the provided list and may appear once at most.
            - Each block has all four lines in the exact order and wording shown above.
            
            IMPORTANT: Output grouped code numbers as plain comma-separated integers with no brackets or extra text (e.g., `Grouped Codes: 1, 4, 7`).\n\n"
            
            ### Initial Codes:\n\n"""
            "```" + full_code_text + "```"
        )
    }

    print(f"[Grouping] Step 1 context chars in prompt: {len(step1_context)}")

    print("[Grouping] Using max_tokens=9000")
    response_ai = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_message, user_message],
        response_format={"type": "text"},
        temperature=0.5,
        max_tokens=9000
    )

    grouped_codes_output = response_ai.choices[0].message.content.strip()

    # Save the raw grouped final codes output to a new file
    grouped_codes_output_path = "New New Inductive Results/Grouped_Codes_Raw.txt"
    with open(grouped_codes_output_path, "w", encoding="utf-8") as f:
        f.write(grouped_codes_output)
    print(f"Grouped final codes raw output saved to {grouped_codes_output_path}")

    # Save grouped codes to final_codebook.xlsx (immediately after sampling)
    final_codebook_xlsx_path = "New New Inductive Results/final_codebook.xlsx"
    # Parse and save in DataFrame: Final Code, Definition, Grouped Codes, Explanation
    final_code_rows = []
    code_sections = re.split(r"\*\*Code:", grouped_codes_output)
    for section in code_sections:
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        code_name = lines[0].strip() if lines else ""
        definition = ""
        grouped_codes = ""
        explanation = ""
        for line in lines:
            if line.strip().startswith("**Definition:**"):
                definition = line.replace("**Definition:**", "").strip()
            elif line.lower().startswith("grouped codes:"):
                grouped_text = line.split(":", 1)[1].strip()
                grouped_codes = normalize_grouped_codes(grouped_text)
            elif line.lower().startswith("explanation:"):
                explanation = line.split(":", 1)[1].strip()
        final_code_rows.append({
            "Final Code": code_name,
            "Definition": definition,
            "Grouped Codes": grouped_codes,
            "Explanation": explanation
        })
    final_codebook_df = pd.DataFrame(final_code_rows)
    # Normalize grouped codes column before saving
    if not final_codebook_df.empty and "Grouped Codes" in final_codebook_df.columns:
        final_codebook_df["Grouped Codes"] = final_codebook_df["Grouped Codes"].apply(normalize_grouped_codes)

    # Prepare DataFrames for additional sheets
    sampled_numbers_path = "New New Inductive Results/sampled_initial_code_numbers.txt"
    refinement_log_path = "New New Inductive Results/refinement_log.csv"
    sampled_code_tracking_df = None
    refinement_log_df = None
    # Sampled code tracking
    if os.path.exists(sampled_numbers_path):
        with open(sampled_numbers_path, "r") as f:
            raw = f.read()
            numbers = re.split(r"[,\n]", raw)
            sampled_numbers = []
            for n in numbers:
                n = n.strip()
                if n and n.isdigit():
                    sampled_numbers.append(int(n))
        # Find grouped numbers in this sample
        grouped_flags = []
        grouped_numbers = set()
        if not final_codebook_df.empty and "Grouped Codes" in final_codebook_df.columns:
            for val in final_codebook_df["Grouped Codes"].dropna():
                val_str = str(val)
                grouped_numbers.update(
                    int(num) for num in re.findall(r"\b\d+\b", val_str)
                )
        grouped_flags = [num in grouped_numbers for num in sampled_numbers]
        sampled_code_tracking_df = pd.DataFrame({
            "Sampled Code Numbers": sampled_numbers,
            "Grouped in Initial Prompt": grouped_flags
        })
        missing_sampled = [num for num, flag in zip(sampled_numbers, grouped_flags) if not flag]
        missing_sampled_df = pd.DataFrame({"Missing Sampled Code Numbers": missing_sampled})
    else:
        sampled_code_tracking_df = pd.DataFrame()
        missing_sampled_df = pd.DataFrame()
    # Refinement log as missing initial codes
    if os.path.exists(refinement_log_path):
        refinement_log_df = pd.read_csv(refinement_log_path)
    else:
        refinement_log_df = pd.DataFrame()
    # Save to Excel using pd.ExcelWriter — write all tracking sheets always, plus sampled codes
    with pd.ExcelWriter(final_codebook_xlsx_path, engine="xlsxwriter") as writer:
        # Main sheet
        final_codebook_df.to_excel(writer, sheet_name="Final Codebook", index=False)

        # 1) Sampled Codes sheet (full records)
        try:
            sampled_codes_full_df = pd.read_csv("New New Inductive Results/initial_code_reference_sampled.csv")
        except Exception:
            sampled_codes_full_df = pd.DataFrame(columns=["Number", "Initial Code", "Definition"])
        sampled_codes_full_df.to_excel(writer, sheet_name="Sampled Codes", index=False)

        # 2) Sampled Code Numbers sheet (numbers + grouped flag) — write even if empty
        if sampled_code_tracking_df is None:
            sampled_code_tracking_df = pd.DataFrame(columns=["Sampled Code Numbers", "Grouped in Initial Prompt"])
        sampled_code_tracking_df.to_excel(writer, sheet_name="Sampled Code Numbers", index=False)

        # 3) Missing Sampled Codes sheet — write even if empty
        if 'missing_sampled_df' not in locals() or missing_sampled_df is None:
            missing_sampled_df = pd.DataFrame({"Missing Sampled Code Numbers": []})
        missing_sampled_df.to_excel(writer, sheet_name="Missing Sampled Codes", index=False)

        # 4) Refinement log if present
        if refinement_log_df is not None and not refinement_log_df.empty:
            refinement_log_df.to_excel(writer, sheet_name="Missing Initial Codes", index=False)

        # 5) Initial Codes reference (full list)
        try:
            initial_codes_ref_df = pd.read_csv("New New Inductive Results/initial_code_reference.csv")
        except Exception:
            initial_codes_ref_df = pd.DataFrame(columns=["Number", "Initial Code", "Definition"])
        initial_codes_ref_df.to_excel(writer, sheet_name="Initial Codes", index=False)
    print(f"Initial final codes saved to {final_codebook_xlsx_path}")
    return grouped_codes_output


# --- Function to get missing codes not yet grouped in final_codebook.csv ---
def get_missing_codes():
    final_codebook_xlsx_path = "New New Inductive Results/final_codebook.xlsx"
    if not os.path.exists(final_codebook_xlsx_path):
        print("final_codebook.xlsx not found.")
        return []
    # Load all initial codes
    initial_code_reference_path = "New New Inductive Results/initial_code_reference.csv"
    if not os.path.exists(initial_code_reference_path):
        print("initial_code_reference.csv not found.")
        return []
    initial_codes_df = pd.read_csv(initial_code_reference_path)
    all_code_numbers = set(initial_codes_df["Number"])
    # Load all grouped codes from final_codebook.xlsx, sheet "Final Codebook"
    final_df = pd.read_excel(final_codebook_xlsx_path, sheet_name="Final Codebook")
    grouped = final_df["Grouped Codes"].dropna().astype(str)
    grouped_numbers = set()
    for val in grouped:
        for num in re.findall(r"\b\d+\b", val):
            grouped_numbers.add(int(num))
    missing = sorted(list(all_code_numbers - grouped_numbers))
    print(f"Identified {len(missing)} missing codes not yet grouped in final_codebook.xlsx.")
    return missing


# === New Function: Codebook Refinement for Unused Codes ===
def run_codebook_refinement(existing_grouped_numbers=None):
    """
    Reads the previously saved Grouped_Codes_Raw.txt, extracts used code numbers,
    compares to all code numbers, and generates a prompt for the AI to assign each unused code
    to an existing final code or a new final code.
    Maintains a persistent set of all grouped code numbers across iterations.
    Args:
        existing_grouped_numbers (set or None): The set of all grouped code numbers from previous iterations.
    Returns:
        still_missing (set): Set of code numbers still missing after refinement.
        combined_grouped_code_numbers (set): Updated set of all grouped code numbers found so far.
    """
    # Use new get_missing_codes function to determine missing codes
    missing_sorted = get_missing_codes()
    if not missing_sorted:
        print("No unused codes to refine. All codes are accounted for.")
        return set(), set()

    # Prepare mapping for numbers to code/definition, with type check
    number_to_code = {item["number"]: item["code"] for item in code_entries if isinstance(item, dict)}
    number_to_def = {item["number"]: item["definition"] for item in code_entries if isinstance(item, dict)}

    # Load current final_codebook.xlsx as reference for grouping
    final_codebook_xlsx_path = "New New Inductive Results/final_codebook.xlsx"
    final_df = pd.read_excel(final_codebook_xlsx_path, sheet_name="Final Codebook")
    # Row numbers for user selection (1-based), only show final code name and definition
    grouped_summary = [
        f"Row {idx+1}: Code: {row['Final Code']}\nDefinition: {row['Definition']}"
        for idx, row in final_df.iterrows()
    ]
    grouped_codes_reference = "\n\n".join(grouped_summary)

    # In-memory DataFrame for updating
    in_memory_final_df = final_df.copy()

    # === Create/initialize refinement log file ===
    refinement_log_path = "New New Inductive Results/refinement_log.csv"
    if not os.path.exists(refinement_log_path):
        with open(refinement_log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Initial Code Number", "Action", "Target Final Code Row or New Code Name"])

    refinement_outputs = []
    refinement_master_path = "New New Inductive Results/Grouped_Codes_Refinement.txt"
    open(refinement_master_path, "a", encoding="utf-8").close()
    for num in missing_sorted:
        code = number_to_code.get(num, "")
        definition = number_to_def.get(num, "")
        unused_codes_text = f"{num}. Code: {code}\nDefinition: {definition}"
        current_grouped_codes_reference = grouped_codes_reference
        prompt = (
            "You are reviewing ONE initial code and its definition, and the list of existing final codes.\n\n"
            "Your task: Decide between exactly two options:\n"
            "A) Assign this initial code to an existing final code (if it clearly fits), OR\n"
            "B) Create ONE new final code (ONLY if it is clearly not covered by ANY existing final code definition).\n"
            "If uncertain after comparing definitions, **assign** to the closest existing final code.\n\n"
            "You must return EXACTLY ONE of the two formats below and NOTHING ELSE.\n\n"
            "FORMAT 1 — Assign to existing (single line):\n"
            "assigned to existing final code ##\n\n"
            "FORMAT 2 — Propose ONE new final code (exactly 4 lines):\n"
            "**Code: [Final Code Name]**  \n"
            "**Definition:** [Concise, concrete definition]  \n"
            "Grouped Codes: [number]  \n"
            "Explanation: [Which existing final code was closest and why it did not apply]\n\n"
            "EXAMPLES (good):\n"
            "assigned to existing final code 12\n\n"
            "**Code: Clarifying Follow-up Questions**  \n"
            "**Definition:** Student explicitly asks targeted follow-up questions to refine the differential.  \n"
            "Grouped Codes: 57  \n"
            "Explanation: Closest was row 9 (History Taking), but that focuses on breadth of initial questions rather than targeted follow-ups.\n\n"
            "BAD EXAMPLE (do NOT do this):\n"
            "“It might be row 12 or 14, and we could also create a new code called X…”  ← Not allowed. You must choose one option.\n\n"
            "REMINDERS\n"
            "- Do not alter existing final code names/definitions/grouped codes.\n"
            "- Use only plain integers for grouped codes (no brackets).\n"
            "- Return exactly one of the two formats; no extra commentary.\n\n"
            "- Consider ALL existing final codes before deciding to create a new one.\n\n"
            "#### Initial Code:\n"
            f"{unused_codes_text}\n\n"
            "#### Existing Final Codes (with row numbers):\n"
            f"{current_grouped_codes_reference}\n"
        )
        system_message = {
            "role": "system",
            "content": (
                "You are an expert qualitative researcher (medical education) applying Braun & Clarke’s (2006) approach. "
                "You have completed familiarization and initial coding. The goal now is to refine **final codes**. Take into consideration implicit meaning in both initial and final code names and definitions when making your decisions.\n\n"
                "DEFINITIONS\n"
                "- A **final code** is a short, specific, concrete label that captures an observable idea explicitly present in the data. "
                "It is not interpretive or thematic; it is precise and descriptive.\n"
                "- A **theme** is broader and interpretive and is NOT what you should produce.\n\n"
                "DECISION RULES (binary)\n"
                "1) If the initial code clearly fits an existing final code’s definition, **assign it to that final code**.\n"
                "2) Create a **new final code** only if the concept is **clearly not** captured by ANY existing final code definition. Consider ALL existing final codes in this step.\n"
                "If uncertain after comparison, **assign to the closest existing final code** (conservative default).\n\n"
                "CONSTRAINTS\n"
                "- Do NOT modify or remove any existing final code names, definitions, or grouped codes.\n"
                "- You may only (a) append the current initial code number to one existing final code, or (b) add one new final code.\n"
                "- Output must match EXACTLY one of the two formats below and NOTHING ELSE.\n"
                "- When you include grouped code numbers, use plain comma-separated integers with no brackets or extra text.\n\n"
                "STEP 1 CONTEXT (high-level familiarization summary used to ground decisions)\n"
                f"{step1_context}\n"
            )
        }
        user_message = {
            "role": "user",
            "content": prompt
        }
        print(f"[Refinement] Using max_tokens=1500 for code {num}")
        response_ai = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message, user_message],
            response_format={"type": "text"},
            temperature=0.2,
            max_tokens=1500
        )
        response = response_ai.choices[0].message.content.strip()
        with open(refinement_master_path, "a", encoding="utf-8") as file:
            file.write(f"\n\n### Refinement for Code {num}\n")
            file.write(response.strip())
        print(f"Refinement result for code {num} saved to refinement history.")
        refinement_outputs.append(response)

        # Prepare a per-code debug directory
        debug_dir = os.path.join("New New Inductive Results", "Refinement_History")
        os.makedirs(debug_dir, exist_ok=True)
        debug_txt_path = os.path.join(debug_dir, f"debug_code_{num}.txt")
        with open(debug_txt_path, "w", encoding="utf-8") as dbg:
            dbg.write(response)

        # Try to parse the response; if it doesn't match either pattern, retry once with a format-only reminder
        def parse_response(text: str):
            m_asg = re.search(
                r"assigned\s+to\s+existing\s+final\s+code\s*(?:row\s*)?(\d+)\b",
                text.strip(),
                re.IGNORECASE,
            )
            m_new = re.search(
                r"\*\*Code:\s*(.+?)\*\*\s*[\r\n]+"
                r"\*\*Definition:\*\*\s*(.+?)\s*[\r\n]+"
                r"Grouped Codes:\s*([^\n\r]+)\s*[\r\n]+"
                r"Explanation:\s*(.+)",
                text.strip(),
                re.IGNORECASE | re.DOTALL,
            )
            return m_asg, m_new

        m_assign, m_new_code = parse_response(response)

        if not m_assign and not m_new_code:
            # Retry once with a minimal, strict reminder
            reminder_user = {
                "role": "user",
                "content": (
                    "Your previous reply did not match the required format. Respond with EXACTLY one of the two formats and nothing else.\n\n"
                    "If assigning: `assigned to existing final code ##` (single line).\n"
                    "If new: the 4-line block with Code / Definition / Grouped Codes / Explanation."
                ),
            }
            response_ai_retry = client.chat.completions.create(
                model="gpt-4o",
                messages=[system_message, user_message, reminder_user],
                response_format={"type": "text"},
                temperature=0.2,
                max_tokens=1500,
            )
            response_retry = response_ai_retry.choices[0].message.content.strip()
            with open(debug_txt_path, "a", encoding="utf-8") as dbg:
                dbg.write("\n\n--- RETRY ---\n\n")
                dbg.write(response_retry)
            m_assign, m_new_code = parse_response(response_retry)
            if m_assign or m_new_code:
                response = response_retry  # use retry content going forward

        if m_assign:
            assigned_row_num = int(m_assign.group(1)) - 1  # zero-based
            if 0 <= assigned_row_num < len(in_memory_final_df):
                grouped_codes_val = str(in_memory_final_df.iloc[assigned_row_num]["Grouped Codes"])  # may be NaN/float
                # Normalize and update grouped codes
                existing_norm = normalize_grouped_codes(grouped_codes_val)
                candidate = normalize_grouped_codes(f"{existing_norm},{num}")
                in_memory_final_df.at[assigned_row_num, "Grouped Codes"] = candidate
            # Log this grouping to refinement_log.csv
            with open(refinement_log_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([num, "Grouped into existing", f"Row {assigned_row_num + 1}"])
        elif m_new_code:
            new_final_code_name = m_new_code.group(1).strip()
            new_final_code_def = m_new_code.group(2).strip()
            grouped_codes_for_new = normalize_grouped_codes(m_new_code.group(3))
            explanation_for_new = m_new_code.group(4).strip()
            new_row = {
                "Final Code": new_final_code_name,
                "Definition": new_final_code_def,
                "Grouped Codes": grouped_codes_for_new,
                "Explanation": explanation_for_new,
            }
            in_memory_final_df = pd.concat([in_memory_final_df, pd.DataFrame([new_row])], ignore_index=True)
            with open(refinement_log_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([num, "Created new final code", new_final_code_name])
        else:
            print(f"Raw unparsed response for code {num}: {response}")

        # Normalize grouped codes column before saving
        if "Grouped Codes" in in_memory_final_df.columns:
            in_memory_final_df["Grouped Codes"] = in_memory_final_df["Grouped Codes"].apply(normalize_grouped_codes)
        # Save updated to final_codebook.xlsx after each refinement
        with pd.ExcelWriter(final_codebook_xlsx_path, engine="xlsxwriter") as writer:
            # Final Codebook
            in_memory_final_df.to_excel(writer, sheet_name="Final Codebook", index=False)

            # Sampled Codes sheet
            try:
                sampled_codes_full_df = pd.read_csv("New New Inductive Results/initial_code_reference_sampled.csv")
            except Exception:
                sampled_codes_full_df = pd.DataFrame(columns=["Number", "Initial Code", "Definition"])
            sampled_codes_full_df.to_excel(writer, sheet_name="Sampled Codes", index=False)

            # Refinement log
            if os.path.exists(refinement_log_path):
                refinement_log_df = pd.read_csv(refinement_log_path)
            else:
                refinement_log_df = pd.DataFrame()
            if not refinement_log_df.empty:
                refinement_log_df.to_excel(writer, sheet_name="Missing Initial Codes", index=False)

            # Initial Codes reference (full list)
            try:
                initial_codes_ref_df = pd.read_csv("New New Inductive Results/initial_code_reference.csv")
            except Exception:
                initial_codes_ref_df = pd.DataFrame(columns=["Number", "Initial Code", "Definition"])
            initial_codes_ref_df.to_excel(writer, sheet_name="Initial Codes", index=False)
        if m_assign:
            print(f"✅ Code {num} grouped into existing final code (row {assigned_row_num + 1}).")
        elif m_new_code:
            print(f"🆕 New final code generated for initial code {num}.")
        else:
            print(f"⚠️ No valid assignment found for code {num}.")
        # Update grouped_summary for next prompt
        grouped_summary = [
            f"Row {idx+1}: Code: {row['Final Code']}\nDefinition: {row['Definition']}"
            for idx, row in in_memory_final_df.iterrows()
        ]
        grouped_codes_reference = "\n\n".join(grouped_summary)

    # After all refinements, check for missing codes again
    still_missing = get_missing_codes()
    if still_missing:
        print(f"⚠️ Still missing code numbers after refinement: {still_missing}")
    else:
        print("✅ All code numbers are now accounted for after refinement.")
    return set(still_missing), set()

def run_quote_selection():
    # Load the final codes from final_codebook.xlsx
    final_codebook_xlsx_path = "New New Inductive Results/final_codebook.xlsx"
    if not os.path.exists(final_codebook_xlsx_path):
        print("❗ final_codebook.xlsx not found. Skipping quote selection.")
        return
    final_df = pd.read_excel(final_codebook_xlsx_path, sheet_name="Final Codebook")

    # Load transcript data
    transcript_df = pd.read_csv("New New Inductive Results/Transcript_Text_Only.csv")
    if "Text" not in transcript_df.columns:
        raise KeyError("Transcript file is missing the 'Text' column.")
    all_text_blocks = transcript_df["Text"].dropna().tolist()
    transcript_reference = "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(all_text_blocks)])

    # System message for quote selection
    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant trained to conduct rigorous qualitative analysis. "
            "Your task is to identify the most representative student quote from the transcript for a given final code and its definition. "
            "You must give EQUAL consideration to every student quote before making your selection."
        )
    }

    # Iterate through each final code
    updated_rows = []
    for idx, row in final_df.iterrows():
        # Flexible lookup for code name column
        code_name_col = next((col for col in row.index if col.lower() in ["final code", "final code name"]), None)
        code_name = row[code_name_col] if code_name_col else "Unknown"
        if code_name_col is None:
            print("⚠️ No valid column found for final code name. Defaulting to 'Unknown'.")
        else:
            print(f"Using final code name: {code_name_col}")

        # Flexible lookup for definition column
        definition_col_name = next((col for col in row.index if col.lower() in ["definition", "final code definition"]), None)
        definition = row[definition_col_name] if definition_col_name else "Unknown"
        if definition_col_name is None:
            print("⚠️ No valid column found for final code definition. Defaulting to 'Unknown'.")
        else:
            print(f"Using final code definition column: {definition_col_name}")

        print(f"Selecting quote for: {code_name}")

        user_message = {
            "role": "user",
            "content": (
                f"### Final Code and Definition\n"
                f"**Code:** {code_name}\n"
                f"**Definition:** {definition}\n\n"
                "You must select a short quote (1–2 sentences maximum) from the transcript that specifically and clearly reflects the final code and its definition. \n"
                "Give EQUAL consideration to every student quote before making your selection.\n\n"
                "### Instructions for Quote Selection:\n"
                "Only include the relevant portion of the student response that directly pertains to the final code — do not return the entire student response.\n\n"
                "Return only the selected excerpt — no commentary or headers.\n"
                "At the end of the quote, include the transcript block number from which it was taken in the format: (ID: ##)\n\n"
                f"### Transcript:\n{transcript_reference}"
            )
        }

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message, user_message],
            response_format={"type": "text"},
            temperature=0.5,
            max_tokens=300
        )
        quote = response.choices[0].message.content.strip()

        # Extract the ID tag and clean the quote
        import re
        match = re.search(r"\(ID:\s*(\d+)\)", quote)
        if match:
            source_row = int(match.group(1))
            quote = re.sub(r"\(ID:\s*\d+\)", "", quote).strip()  # Remove the ID tag from the quote
        else:
            source_row = ""

        # Set the quote directly in the "Example" column
        updated_row = row.to_dict()
        updated_row["Example"] = quote
        updated_row["Example Source Row"] = source_row
        updated_rows.append(updated_row)

    # Save updated final codebook
    updated_df = pd.DataFrame(updated_rows)
    # Normalize grouped codes column before saving
    if "Grouped Codes" in updated_df.columns:
        updated_df["Grouped Codes"] = updated_df["Grouped Codes"].apply(normalize_grouped_codes)
    # Add Grouped Code Names column to updated_df
    initial_code_reference_path = "New New Inductive Results/initial_code_reference.csv"
    if os.path.exists(initial_code_reference_path):
        initial_df = pd.read_csv(initial_code_reference_path)
        number_to_code = dict(zip(initial_df["Number"], initial_df["Initial Code"]))
        grouped_names = []
        for grouped in updated_df["Grouped Codes"]:
            if pd.isna(grouped):
                grouped_names.append("")
                continue
            numbers = [int(s) for s in re.findall(r"\b\d+\b", str(grouped))]
            names = [number_to_code.get(n, "") for n in numbers]
            grouped_names.append(", ".join(filter(None, names)))
        updated_df["Grouped Code Names"] = grouped_names
    # Save back to Excel
    with pd.ExcelWriter(final_codebook_xlsx_path, engine="xlsxwriter") as writer:
        # Updated Final Codebook with Examples
        updated_df.to_excel(writer, sheet_name="Final Codebook", index=False)

        # Sampled Codes (full rows)
        try:
            sampled_codes_full_df = pd.read_csv("New New Inductive Results/initial_code_reference_sampled.csv")
        except Exception:
            sampled_codes_full_df = pd.DataFrame(columns=["Number", "Initial Code", "Definition"])
        sampled_codes_full_df.to_excel(writer, sheet_name="Sampled Codes", index=False)

        # Sampled Code Numbers and Missing Sampled Codes (always write)
        sampled_numbers_path = "New New Inductive Results/sampled_initial_code_numbers.txt"
        if os.path.exists(sampled_numbers_path):
            with open(sampled_numbers_path, "r") as f:
                raw = f.read()
                numbers = re.split(r"[,\n]", raw)
                sampled_numbers = []
                for n in numbers:
                    n = n.strip()
                    if n and n.isdigit():
                        sampled_numbers.append(int(n))
        else:
            sampled_numbers = []

        grouped_numbers = set()
        if "Grouped Codes" in updated_df.columns:
            for val in updated_df["Grouped Codes"].dropna():
                norm = normalize_grouped_codes(val)
                grouped_numbers.update(int(num) for num in re.findall(r"\b\d+\b", norm))

        sampled_code_tracking_df = pd.DataFrame({
            "Sampled Code Numbers": sampled_numbers,
            "Grouped in Initial Prompt": [num in grouped_numbers for num in sampled_numbers]
        })
        missing_sampled = [num for num in sampled_numbers if num not in grouped_numbers]
        missing_sampled_df = pd.DataFrame({"Missing Sampled Code Numbers": missing_sampled})

        sampled_code_tracking_df.to_excel(writer, sheet_name="Sampled Code Numbers", index=False)
        missing_sampled_df.to_excel(writer, sheet_name="Missing Sampled Codes", index=False)

        # Refinement log
        refinement_log_path = "New New Inductive Results/refinement_log.csv"
        if os.path.exists(refinement_log_path):
            refinement_log_df = pd.read_csv(refinement_log_path)
            if not refinement_log_df.empty:
                refinement_log_df.to_excel(writer, sheet_name="Missing Initial Codes", index=False)

        # Initial Codes reference (full list)
        try:
            initial_codes_ref_df = pd.read_csv("New New Inductive Results/initial_code_reference.csv")
        except Exception:
            initial_codes_ref_df = pd.DataFrame(columns=["Number", "Initial Code", "Definition"])
        initial_codes_ref_df.to_excel(writer, sheet_name="Initial Codes", index=False)
    print("✅ Quote selection complete and saved to final_codebook.xlsx")




if __name__ == "__main__":
    # Set these flags to control which part of the pipeline to run
    RUN_INITIAL_CODEBOOK_GENERATION = True
    RUN_CODEBOOK_REFINEMENT = True
    RUN_QUOTE_SELECTION = True

    # --- Reset refinement log at the start of every run so the "Missing Initial Codes" sheet doesn't carry over old data ---
    refinement_log_path = "New New Inductive Results/refinement_log.csv"
    try:
        with open(refinement_log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Initial Code Number", "Action", "Target Final Code Row or New Code Name"])
        print("🧹 Cleared previous refinement log (New New Inductive Results/refinement_log.csv).")
    except Exception as e:
        print(f"⚠️ Could not reset refinement log: {e}")

    if RUN_INITIAL_CODEBOOK_GENERATION:
        # If you want to save the codebook output, use save_final_codebook_output
        run_code_grouping()

    if RUN_CODEBOOK_REFINEMENT:
        still_missing, _ = run_codebook_refinement()
        while still_missing:
            print("🔁 Rerunning codebook refinement due to remaining missing codes...")
            still_missing, _ = run_codebook_refinement()
        final_check_missing = get_missing_codes()
        if final_check_missing:
            print(f"❗ Final check — still missing code numbers: {final_check_missing}")
        else:
            print("✅ Final check passed — all initial codes are now grouped into final codes.")

    if RUN_QUOTE_SELECTION:
        run_quote_selection()
