import argparse
from pathlib import Path
from typing import Iterable

try:
    import sacrebleu
    import torch
    from striprtf.striprtf import rtf_to_text
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as exc:
    missing_dependency = exc.name or "required package"
    raise SystemExit(
        "Missing dependency: "
        f"{missing_dependency}. Install the required packages with "
        "`pip install torch transformers sentencepiece striprtf sacrebleu`."
    ) from exc


MODEL_NAME = "Helsinki-NLP/opus-mt-bn-en"
DEFAULT_INPUT = Path("./data/input.rtf")
DEFAULT_OUTPUT = Path("./output.txt")
DEFAULT_EVAL_FILE = Path("./data/output.rtf")
DEFAULT_OUTPUT_HEADER = "# reference_english.txt"
DEFAULT_REFERENCE = Path("./data/output.rtf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Bengali text to English with Helsinki-NLP/opus-mt-bn-en."
    )
    parser.add_argument(
        "--text",
        help="Direct Bengali text to translate. If omitted, the script reads from --input-file.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the input text or RTF file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to save the translated output. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of lines to translate per batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per line.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. Default: auto",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help=f"File to evaluate with sacrebleu. Default: {DEFAULT_EVAL_FILE}",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"Reference file for BLEU evaluation. Default: {DEFAULT_REFERENCE}",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate translations with sacrebleu after generation.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip generation and evaluate an existing hypothesis file.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return choice


def read_input(text: str | None, input_file: Path) -> tuple[str | None, str]:
    if text:
        return None, text

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_file.suffix.lower() == ".rtf":
        with input_file.open("r", encoding="utf-8", errors="ignore") as source_file:
            source_text = rtf_to_text(source_file.read())
    else:
        source_text = input_file.read_text(encoding="utf-8")

    return extract_header_and_body(source_text)


def split_segments(text: str) -> list[str]:
    segments = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("#"):
            continue
        segments.append(cleaned)
    return segments


def extract_header_and_body(text: str) -> tuple[str | None, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, ""

    if lines[0].startswith("#"):
        return lines[0], "\n".join(lines[1:])

    return None, "\n".join(lines)


def read_segments_from_file(path: Path) -> list[str]:
    _, text = read_input(None, path)
    segments = split_segments(text)
    if not segments:
        raise ValueError(f"No segments found in file: {path}")
    return segments


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def translate_segments(
    segments: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    translations: list[str] = []

    for batch in batched(segments, batch_size):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
            )

        translations.extend(
            tokenizer.batch_decode(generated, skip_special_tokens=True)
        )

    return translations


def rtf_escape(text: str) -> str:
    escaped_parts = []
    for char in text:
        if char == "\\":
            escaped_parts.append("\\\\")
        elif char == "{":
            escaped_parts.append("\\{")
        elif char == "}":
            escaped_parts.append("\\}")
        elif char == "\n":
            escaped_parts.append("\\par\n")
        elif ord(char) > 127:
            escaped_parts.append(f"\\u{ord(char)}?")
        else:
            escaped_parts.append(char)
    return "".join(escaped_parts)


def write_output(
    output_file: Path | None,
    translated_text: str,
    header: str | None = None,
) -> None:
    if output_file is None:
        print(translated_text)
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.suffix.lower() == ".rtf":
        body = translated_text
        if header:
            body = f"{header}\n\n{translated_text}"
        rtf_content = (
            "{\\rtf1\\ansi\\deff0\n"
            "{\\fonttbl{\\f0\\fswiss Helvetica;}}\n"
            "\\f0\\fs24 "
            f"{rtf_escape(body)}"
            "\n}"
        )
        output_file.write_text(rtf_content, encoding="utf-8")
    else:
        output_file.write_text(translated_text + "\n", encoding="utf-8")


def evaluate_translations(hypothesis_file: Path, reference_file: Path) -> sacrebleu.metrics.bleu.BLEUScore:
    hypotheses = read_segments_from_file(hypothesis_file)
    references = read_segments_from_file(reference_file)

    if len(hypotheses) != len(references):
        raise ValueError(
            "Hypothesis/reference segment count mismatch: "
            f"{len(hypotheses)} vs {len(references)}."
        )

    return sacrebleu.corpus_bleu(hypotheses, [references])


def main() -> None:
    args = parse_args()

    if args.evaluate_only:
        bleu = evaluate_translations(args.eval_file, args.reference_file)
        print(f"BLEU = {bleu.score:.2f}")
        print(bleu.format())
        return

    device = resolve_device(args.device)
    input_header, source_text = read_input(args.text, args.input_file)
    segments = split_segments(source_text)

    if not segments:
        raise ValueError("No translatable content was found in the provided input.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    translations = translate_segments(
        segments=segments,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    translated_text = "\n\n".join(translations)
    output_header = (
        DEFAULT_OUTPUT_HEADER
        if input_header and args.output_file.suffix.lower() == ".rtf"
        else None
    )
    write_output(args.output_file, translated_text, header=output_header)
    print(translated_text)

    if args.evaluate:
        bleu = evaluate_translations(args.eval_file, args.reference_file)
        print(f"\nBLEU = {bleu.score:.2f}")
        print(bleu.format())


if __name__ == "__main__":
    main()
