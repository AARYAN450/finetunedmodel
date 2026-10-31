"""Drop the results table into README.md. Run after baseline.py / train.py.

Usage: python write_report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
MARKER = "<!-- RESULTS_TABLE -->"


def load(name):
    path = ROOT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    baseline = load("results_baseline.json")
    roberta = load("results_roberta.json")

    lines = [
        "| model | precision | recall | F1 |",
        "|---|---|---|---|",
    ]
    if baseline:
        lines.append(
            f"| TF-IDF + LogisticRegression (baseline) | {baseline['precision']:.3f} "
            f"| {baseline['recall']:.3f} | **{baseline['f1']:.3f}** |"
        )
    if roberta:
        at = roberta["at_0.5"]
        lines.append(
            f"| `roberta-base` fine-tuned, threshold 0.50 | {at['precision']:.3f} "
            f"| {at['recall']:.3f} | **{at['f1']:.3f}** |"
        )
        best = roberta.get("best_threshold")
        if best and abs(best["threshold"] - 0.5) > 1e-6:
            lines.append(
                f"| `roberta-base`, tuned threshold {best['threshold']:.2f} "
                f"| {best['precision']:.3f} | {best['recall']:.3f} "
                f"| **{best['f1']:.3f}** |"
            )
        lines.append("")
        lines.append(
            f"Trained on {'GPU' if roberta['device'] == 'cuda' else 'CPU'} in "
            f"{roberta['train_seconds']:.0f}s "
            f"({roberta['epochs']} epochs, batch {roberta['batch_size']}, "
            f"max_len {roberta['max_len']}, lr {roberta['lr']})."
        )
    else:
        lines.append("")
        lines.append("_RoBERTa results not available._")

    if baseline and roberta:
        delta = roberta["at_0.5"]["f1"] - baseline["f1"]
        word = "beats" if delta > 0.01 else ("matches" if abs(delta) <= 0.01 else "loses to")
        lines.append("")
        lines.append(
            f"At threshold 0.50 the fine-tuned model **{word}** the baseline by "
            f"{delta:+.3f} F1 on the action-item class."
        )

    test_n = (roberta or baseline or {}).get("test_rows")
    test_pos = (roberta or baseline or {}).get("test_positives")
    if test_n:
        lines.append("")
        lines.append(
            f"Test set: {test_n} sentences, {test_pos} action items "
            f"(stratified 20%, seed 42, identical rows for both models)."
        )

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    table = "\n".join(lines)
    if MARKER not in text:
        # The Results section has already been written and hand-edited;
        # appending again would duplicate the table.
        print("README.md already has a results section -- not touching it.")
        print("Delete the section and restore the marker to regenerate.\n")
        print(table)
        return
    readme.write_text(text.replace(MARKER, table), encoding="utf-8")
    print("README.md updated:\n")
    print(table)


if __name__ == "__main__":
    main()
