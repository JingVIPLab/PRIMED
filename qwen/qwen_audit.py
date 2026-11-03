import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv


AUDIT_PROMPT = \
"""
Role:
You are a professional auditor for Referring Audio-Visual Segmentation tasks. You possess a deep understanding of how language refers to objects through different modalities—specifically distinguishing between static visual cues and temporal audio cues.

Task:
Your goal is to conduct a rigorous audit of a Referring Audio-Visual Segmentation dataset.
For each entry, you must perform a cross-modal logic verification:
· Semantic Analysis: Deconstruct the input text reference to identify the primary and secondary cues required for target localization.
· Consistency Check: Compare your logical deduction against the current label provided in the input.
· Validation & Correction: Determine if the label is technically accurate based on the grounding requirements. If the label is inaccurate, you must flag it as "Incorrect" and provide the scientifically accurate modality and a logic-driven correction.

Guidelines:
· Visual: The object is identified solely by physical attributes (color, shape, position, or relationships).
· Audio: The object is identified solely by its temporal sequence or acoustic properties within the audio stream.
· Audio-Visual: The localization simultaneously requires both modalities to distinguish from other candidates.
· Output Format: {Status}. Correct Modality: {Modality} (if needed). (Explanation: {Reasoning})

Examples:
Input1: The object that making a sound after the flute. (Current Label: Audio)
Output1: Correct. (Explanation: The target is identified exclusively by its relative temporal position in the audio track following a specific instrument's sound.)
Input2: The object being held by the woman. (Current Label: Audio-Visual)
Output2: Incorrect. (Explanation: Identification depends entirely on the object's appearance and its static spatial contact with a person, independent of any acoustic information.)
"""

MODALITIES = {label.lower(): label for label in ("Visual", "Audio", "Audio-Visual")}
LABEL_PATTERN = re.compile(r"^(Audio-Visual|Visual|Audio)(?:\.|\s*$)", re.IGNORECASE)
AUDIT_PATTERN = re.compile(
    r"(?P<status>Correct|Incorrect)\.\s*"
    r"(?:Correct Modality:\s*(?P<modality>Audio-Visual|Visual|Audio)\.\s*)?"
    r"\(Explanation:\s*(?P<explanation>.+)\)",
    re.DOTALL,
)
AUDIT_FIELDS = [
    "annotation_label", "audit_result", "audit_status", "audit_modality",
    "audit_explanation", "audit_error", "audit_attempts", "audit_seconds",
    "review_required",
    "audit_model", "audit_temperature", "audit_prompt_sha256",
]
LEGACY_LABEL_FIELDS = ['label_modality', 'label_explanation', 'label_raw', 'label_error', 'label_attempts', 'label_seconds', 'label_model', 'label_temperature', 'label_prompt_sha256', 'label_base_url']
PROMPT_SHA256 = hashlib.sha256(AUDIT_PROMPT.encode("utf-8")).hexdigest()


def parse_annotation(value):
    """Read only the leading label, never modality words in the explanation."""
    match = LABEL_PATTERN.match(value.strip())
    if not match:
        raise ValueError("The initial annotation has no valid leading modality label.")
    return MODALITIES[match.group(1).lower()]


def parse_audit_response(response, current_label):
    """Validate the paper's response format and reject contradictory decisions."""
    match = AUDIT_PATTERN.fullmatch(response.strip())
    if not match or not match.group("explanation").strip():
        raise ValueError("The audit response does not match the paper's output format.")
    status = match.group("status")
    modality = match.group("modality")
    if status == "Incorrect":
        if modality is None:
            raise ValueError("An Incorrect response must provide Correct Modality.")
        if modality == current_label:
            raise ValueError("An Incorrect response cannot retain the current label.")
    elif modality is not None and modality != current_label:
        raise ValueError("A Correct response cannot change the current label.")
    return status, modality or current_label, match.group("explanation").strip()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"A CSV row does not match the header: {path}")
    return fields, rows


def write_csv(path, fields, rows):
    """Replace a CSV atomically so interrupted runs retain the last checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def state_path(output):
    return output.with_name(output.name + ".state.json")


def write_state(output, fields, rows, metadata_fields):
    """Keep retry and provenance details outside the result CSV."""
    path = state_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump({
                "fields": fields,
                "rows": [{field: row[field] for field in metadata_fields} for row in rows],
            }, stream, ensure_ascii=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore_state(output, fields, rows, metadata_fields, result_field):
    path = state_path(output)
    if not path.exists():
        raise ValueError(f"The resume state file is missing: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(state, dict) or state.get("fields") != fields
        or not isinstance(state.get("rows"), list) or len(state["rows"]) != len(rows)
        or any(not isinstance(item, dict) or set(item) != set(metadata_fields) for item in state["rows"])
        or any(not isinstance(value, str) for item in state["rows"] for value in item.values())
    ):
        raise ValueError(f"The resume state does not match this output: {path}")
    for row, metadata in zip(rows, state["rows"]):
        if row[result_field] == metadata[result_field]:
            row.update(metadata)
    return rows


def save_results(output, review, fields, rows):
    csv_fields = [field for field in fields if field not in AUDIT_FIELDS or field == "audit_result"]
    write_state(output, fields, rows, AUDIT_FIELDS)
    write_csv(output, csv_fields, [{field: row[field] for field in csv_fields} for row in rows])
    write_csv(review, csv_fields, [
        {field: row[field] for field in csv_fields}
        for row in rows if row["review_required"] == "True"
    ])


def prepare_rows(input_path, output_path, resume):
    source_fields, source_rows = read_csv(input_path)
    source_fields = [field for field in source_fields if field not in LEGACY_LABEL_FIELDS]
    source_rows = [{field: row[field] for field in source_fields} for row in source_rows]
    if not {"exp", "api_result"}.issubset(source_fields):
        raise ValueError("The input CSV must contain 'exp' and 'api_result' columns.")
    if set(source_fields) & set(AUDIT_FIELDS):
        raise ValueError("Use the original annotation CSV as input, not an audit output.")
    fields = source_fields + AUDIT_FIELDS
    csv_fields = source_fields + ["audit_result"]
    rows = [dict(row, **dict.fromkeys(AUDIT_FIELDS, "")) for row in source_rows]
    if output_path.exists():
        if not resume:
            raise ValueError("The output already exists. Use --resume or choose a new --output.")
        saved_fields, saved_rows = read_csv(output_path)
        saved_fields = [field for field in saved_fields if field not in LEGACY_LABEL_FIELDS]
        saved_rows = [{field: row[field] for field in saved_fields} for row in saved_rows]
        if saved_fields not in (fields, csv_fields) or len(saved_rows) != len(rows):
            raise ValueError("The saved audit does not match the input CSV.")
        for source, saved in zip(source_rows, saved_rows):
            if any(source[field] != saved[field] for field in source_fields):
                raise ValueError("Input rows changed since the previous run; choose a new output.")
        if saved_fields == fields:
            rows = saved_rows
        else:
            for row, saved in zip(rows, saved_rows):
                row["audit_result"] = saved["audit_result"]
            rows = restore_state(output_path, fields, rows, AUDIT_FIELDS, "audit_result")
    return fields, rows


def already_audited(row):
    try:
        label = parse_annotation(row["api_result"])
        status, modality, explanation = parse_audit_response(row["audit_result"], label)
        return (
            not row["audit_error"] and row["annotation_label"] == label
            and row["audit_status"] == status and row["audit_modality"] == modality
            and row["audit_explanation"] == explanation
            and row["review_required"] == str(status == "Incorrect")
        )
    except ValueError:
        return False


def audit_row(row, client, model, temperature, max_attempts, retry_delay):
    """Retain raw model output separately from script-level errors."""
    row.update(dict.fromkeys(AUDIT_FIELDS, ""))
    row["audit_attempts"] = "0"
    row["review_required"] = "True"
    row["audit_model"] = model
    row["audit_temperature"] = str(temperature)
    row["audit_prompt_sha256"] = PROMPT_SHA256
    started = time.perf_counter()
    try:
        if not row["exp"].strip():
            raise ValueError("The referring expression is empty.")
        row["annotation_label"] = parse_annotation(row["api_result"])
        user_input = f'{row["exp"]} (Current Label: {row["annotation_label"]})'
        for attempt in range(1, max_attempts + 1):
            row["audit_attempts"] = str(attempt)
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": AUDIT_PROMPT},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=temperature,
                )
                row["audit_result"] = completion.choices[0].message.content or ""
                status, modality, explanation = parse_audit_response(
                    row["audit_result"], row["annotation_label"]
                )
                row.update(
                    audit_status=status, audit_modality=modality,
                    audit_explanation=explanation, audit_error="",
                    review_required=str(status == "Incorrect"),
                )
                break
            except Exception as error:
                row["audit_error"] = f"{type(error).__name__}: {error}"
                if attempt < max_attempts:
                    time.sleep(min(retry_delay * 2 ** (attempt - 1), 60))
    except KeyboardInterrupt:
        row["audit_error"] = "Interrupted before the audit completed. Resume to retry this row."
        raise
    except ValueError as error:
        row["audit_error"] = str(error)
    finally:
        row["audit_seconds"] = f"{time.perf_counter() - started:.2f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(r"input/path/to/metadata_annotated.csv"))
    parser.add_argument("--output", type=Path, help="Default: <input_stem>_audited.csv")
    parser.add_argument("--review-output", type=Path, help="Default: <output_stem>_review.csv")
    parser.add_argument("--model", default="qwen3-omni-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0, help="API request timeout in seconds")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", help="Skip successful audits in an existing output")
    parser.add_argument("--limit", type=int, help="Process at most this many pending rows")
    parser.add_argument("--save-every", type=int, default=20, help="Checkpoint every N processed rows (default: 20)")
    args = parser.parse_args()
    if args.max_attempts < 1 or args.timeout <= 0 or args.retry_delay < 0:
        parser.error("Attempts and timeout must be positive; retry delay must be nonnegative.")
    if not 0 <= args.temperature <= 2:
        parser.error("Temperature must be between 0 and 2.")
    if args.limit is not None and args.limit < 1:
        parser.error("Limit must be positive.")
    if args.save_every < 1:
        parser.error("Save interval must be positive.")
    return args


def main():
    args = parse_args()
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    output = args.output or args.input.with_name(args.input.stem + "_audited.csv")
    review = args.review_output or output.with_name(output.stem + "_review.csv")
    paths = [path.resolve() for path in (args.input, output, review)]
    if len(set(paths)) != 3:
        raise ValueError("Input, output, and review output must be different files.")
    fields, rows = prepare_rows(args.input, output, args.resume)
    if review.exists() and not (args.resume and output.exists()):
        raise ValueError("The review output already exists; choose a new --review-output.")

    pending = [
        (index, row) for index, row in enumerate(rows)
        if not (
            already_audited(row) and row["audit_model"] == args.model
            and row["audit_temperature"] == str(args.temperature)
            and row["audit_prompt_sha256"] == PROMPT_SHA256
        )
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
    client = None
    if pending:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Set DASHSCOPE_API_KEY in the script directory's .env file or in the environment.")
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=0,
        )

    print(f"Input: {args.input}")
    print(f"Output: {output}")
    print(f"Review queue: {review}")
    print(f"Rows to process: {len(pending)} / {len(rows)}")
    try:
        for processed, (index, row) in enumerate(pending, 1):
            print(f"Processing row {index + 1}/{len(rows)}: {row['exp']}", flush=True)
            audit_row(row, client, args.model, args.temperature, args.max_attempts, args.retry_delay)
            if processed % args.save_every == 0:
                save_results(output, review, fields, rows)
            if row["audit_error"]:
                print(f"Audit error: {row['audit_error']}", flush=True)
            else:
                print(row["audit_result"], flush=True)
    finally:
        save_results(output, review, fields, rows)
        if client is not None:
            client.close()
    correct = sum(row["audit_status"] == "Correct" for row in rows)
    incorrect = sum(row["audit_status"] == "Incorrect" for row in rows)
    errors = sum(bool(row["audit_error"]) for row in rows)
    print(f"Correct: {correct}; Incorrect: {incorrect}; Errors: {errors}")
    print(f"Pending: {len(rows) - correct - incorrect - errors}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
