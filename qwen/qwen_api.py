import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv


ANNOTATION_PROMPT = """
Role: 
You are an expert for Referring Audio-Visual Segmentation tasks. Your goal is to determine which modalities are required to identify and locate the target object based on a given text reference.

Task:
Analyze the provided text reference and categorize the essential modalities into one of three types: visual, audio, or audio-visual.

Modality Definitions:
1. Visual: The object can be uniquely identified using only visual cues (e.g., color, shape, position, or relationships) without needing to hear any sound.
2. Audio: The object is identified solely through its acoustic characteristics or its temporal position in the audio stream (e.g., the order of sound sources), where visual appearance is secondary to the sound's timing.
3. Audio-Visual: The object is identified by perceiving both visual presence and sound-making activity. This includes:
	· The target is the direct source of the sound.
	· The target is spatially or logically linked to a sound-making event or entity.

Guidelines:
1. Your response must start with one of the three keywords: Visual, Audio, or Audio-Visual.
2. The keyword must be followed by a single sentence to explain the logical necessity of that modality.
3. Output Format: {Modality}. (Explanation: {Reasoning})

Examples:
Input1: The object that making a sound after the flute.
Output1: Audio. (Explanation: The target is identified exclusively by its relative temporal position in the audio track following a specific instrument's sound.)
Input2: The object being held by the woman.
Output2: Visual. (Explanation: Identification depends entirely on the object's appearance and its static spatial contact with a person, independent of any acoustic information.)
Input3: The sounding object near the woman.
Output3: Audio-Visual. (Explanation: Locating the target requires synchronizing its visual proximity to the woman with the confirmation of its active sound-producing state via audio.)
"""

LABEL_PATTERN = re.compile(
    r"(?P<modality>Audio-Visual|Visual|Audio)\.\s*"
    r"\(Explanation:\s*(?P<explanation>.+)\)",
    re.DOTALL,
)
LABEL_FIELDS = [
    "api_result", "label_modality", "label_explanation", "label_raw",
    "label_error", "label_attempts", "label_seconds", "label_model",
    "label_temperature", "label_prompt_sha256", "label_base_url",
]
PROMPT_SHA256 = hashlib.sha256(ANNOTATION_PROMPT.encode("utf-8")).hexdigest()


def parse_annotation_response(response):
    """Parse the leading modality and explanation using the paper's format."""
    match = LABEL_PATTERN.fullmatch(response.strip())
    if not match or not match.group("explanation").strip():
        raise ValueError("The annotation response does not match the paper's output format.")
    return match.group("modality"), match.group("explanation").strip()


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


def save_results(output, fields, rows):
    csv_fields = [field for field in fields if field not in LABEL_FIELDS or field == "api_result"]
    write_state(output, fields, rows, LABEL_FIELDS)
    write_csv(output, csv_fields, [{field: row[field] for field in csv_fields} for row in rows])


def prepare_rows(input_path, output_path, resume):
    source_fields, source_rows = read_csv(input_path)
    if "exp" not in source_fields:
        raise ValueError("The input CSV must contain an 'exp' column.")
    if set(source_fields) & set(LABEL_FIELDS):
        raise ValueError("Use the original metadata CSV as input, not an annotation output.")
    fields = source_fields + LABEL_FIELDS
    csv_fields = source_fields + ["api_result"]
    rows = [dict(row, **dict.fromkeys(LABEL_FIELDS, "")) for row in source_rows]
    if output_path.exists():
        if not resume:
            raise ValueError("The output already exists. Use --resume or choose a new --output.")
        saved_fields, saved_rows = read_csv(output_path)
        if saved_fields not in (fields, csv_fields) or len(saved_rows) != len(rows):
            raise ValueError("The saved annotations do not match the input CSV.")
        for source, saved in zip(source_rows, saved_rows):
            if any(source[field] != saved[field] for field in source_fields):
                raise ValueError("Input rows changed since the previous run; choose a new output.")
        if saved_fields == fields:
            rows = saved_rows
        else:
            for row, saved in zip(rows, saved_rows):
                row["api_result"] = saved["api_result"]
            rows = restore_state(output_path, fields, rows, LABEL_FIELDS, "api_result")
    return fields, rows


def already_annotated(row, model, temperature, base_url):
    try:
        modality, explanation = parse_annotation_response(row["api_result"])
        return (
            not row["label_error"] and row["label_modality"] == modality
            and row["label_explanation"] == explanation
            and row["label_raw"] == row["api_result"]
            and row["label_model"] == model
            and row["label_temperature"] == str(temperature)
            and row["label_prompt_sha256"] == PROMPT_SHA256
            and row["label_base_url"] == base_url
        )
    except ValueError:
        return False


def execute_task(user_input, client, model="qwen3-omni-flash", temperature=0.7):
    """Request a label without constructing a client at import time."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANNOTATION_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=temperature,
    )
    return completion.choices[0].message.content or ""


def annotate_row(row, client, model, temperature, base_url, max_attempts, retry_delay):
    """Store validated results separately from raw responses and request errors."""
    row.update(dict.fromkeys(LABEL_FIELDS, ""))
    row.update(
        api_result="Error", label_attempts="0", label_model=model,
        label_temperature=str(temperature), label_prompt_sha256=PROMPT_SHA256,
        label_base_url=base_url,
    )
    started = time.perf_counter()
    try:
        if not row["exp"].strip():
            raise ValueError("The referring expression is empty.")
        for attempt in range(1, max_attempts + 1):
            row["label_attempts"] = str(attempt)
            try:
                row["label_raw"] = execute_task(row["exp"], client, model, temperature)
                modality, explanation = parse_annotation_response(row["label_raw"])
                row.update(
                    api_result=row["label_raw"], label_modality=modality,
                    label_explanation=explanation, label_error="",
                )
                break
            except Exception as error:
                row["label_error"] = f"{type(error).__name__}: {error}"
                if attempt < max_attempts:
                    time.sleep(min(retry_delay * 2 ** (attempt - 1), 60))
    except KeyboardInterrupt:
        row["label_error"] = "Interrupted before annotation completed. Resume to retry this row."
        raise
    except ValueError as error:
        row["label_error"] = str(error)
    finally:
        row["label_seconds"] = f"{time.perf_counter() - started:.2f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(r"input/path/to/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path(r"output/path/to/metadata_annotated.csv"))
    parser.add_argument("--model", default="qwen3-omni-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=60.0, help="API request timeout in seconds")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", help="Skip successful annotations from an existing output")
    parser.add_argument("--limit", type=int, help="Process at most this many pending rows")
    parser.add_argument("--save-every", type=int, default=20, help="Checkpoint every N processed rows (default: 20)")
    args = parser.parse_args()
    if args.max_attempts < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Attempts and timeout must be positive; timeout must be finite.")
    if not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("Retry delay must be finite and nonnegative.")
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
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files.")
    fields, rows = prepare_rows(args.input, args.output, args.resume)
    pending = [
        (index, row) for index, row in enumerate(rows)
        if not already_annotated(row, args.model, args.temperature, args.base_url)
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
    client = None
    if pending:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set DASHSCOPE_API_KEY in the script directory's .env file or in the environment.")
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=0,
        )

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}; Temperature: {args.temperature}")
    print(f"Rows to process: {len(pending)} / {len(rows)}")
    started = time.perf_counter()
    try:
        for processed, (index, row) in enumerate(pending, 1):
            print(f"Processing row {index + 1}/{len(rows)}: {row['exp']}", flush=True)
            annotate_row(
                row, client, args.model, args.temperature, args.base_url,
                args.max_attempts, args.retry_delay,
            )
            if processed % args.save_every == 0:
                save_results(args.output, fields, rows)
            if row["label_error"]:
                print(f"Annotation error: {row['label_error']}", flush=True)
            else:
                print(row["api_result"], flush=True)
            print(f"Duration: {row['label_seconds']} seconds", flush=True)
    finally:
        try:
            save_results(args.output, fields, rows)
        finally:
            if client is not None:
                client.close()
    successful = sum(bool(row["label_modality"]) and not row["label_error"] for row in rows)
    errors = sum(bool(row["label_error"]) for row in rows)
    print(f"Successful: {successful}; Errors: {errors}; Pending: {len(rows) - successful - errors}")
    duration = time.perf_counter() - started
    print(f"Run duration: {timedelta(seconds=int(duration))} ({duration:.2f} seconds)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
