"""Comparison table + the keyword-mimicry check.

Reads results_baseline.json / results_roberta.json for the headline table, then
re-scores the test set with the fine-tuned model to show where it DISAGREES
with the weak-label rules. Disagreements are the only evidence that the model
learned something beyond the regexes it was trained on.

Usage: python evaluate.py
"""
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data import load_split
from extract import load_model, score
from weak_label import STRONG, TASK_VERB, classify

ROOT = Path(__file__).parent


def load_results(name):
    path = ROOT / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table(baseline, roberta):
    print("=" * 68)
    print("ACTION-ITEM CLASS (positive class) -- identical test split, seed 42")
    print("=" * 68)
    print(f"{'model':<28}{'precision':>11}{'recall':>9}{'f1':>8}")
    print("-" * 68)
    rows = []
    if baseline:
        rows.append(("TF-IDF + LogisticRegression", baseline))
    if roberta:
        rows.append(("roberta-base (fine-tuned)", roberta["at_0.5"]))
        best = roberta.get("best_threshold")
        if best:
            rows.append(
                (f"roberta-base @ thr {best['threshold']}", best)
            )
    for label, metrics in rows:
        print(
            f"{label:<28}{metrics['precision']:>11.3f}"
            f"{metrics['recall']:>9.3f}{metrics['f1']:>8.3f}"
        )
    print("-" * 68)
    if baseline and roberta:
        delta = roberta["at_0.5"]["f1"] - baseline["f1"]
        verdict = "beats" if delta > 0 else ("matches" if abs(delta) < 0.01 else "LOSES TO")
        print(f"roberta-base {verdict} baseline by {delta:+.3f} F1 @0.5")
    print()


def disagreements(limit=12):
    """Sentences where the model and the weak-label rules disagree."""
    _, test_df = load_split()
    tokenizer, model = load_model()
    probs = score(tokenizer, model, list(test_df.sentence))

    model_pos = [p >= 0.5 for p in probs]
    rule_pos = list(test_df.label.astype(bool))

    print("=" * 68)
    print("KEYWORD-MIMICRY CHECK")
    print("=" * 68)
    tn, fp, fn, tp = confusion_matrix(rule_pos, model_pos, labels=[0, 1]).ravel()
    agree = (tp + tn) / len(probs)
    print(f"model/rule agreement on test: {agree:.1%}  "
          f"(model-only +{fp}, rule-only -{fn})\n")

    rows = sorted(
        (
            (probs[i], test_df.sentence[i], rule_pos[i], model_pos[i])
            for i in range(len(probs))
            if rule_pos[i] != model_pos[i]
        ),
        key=lambda r: -r[0],
    )

    print(f"-- model says ACTION ITEM, rules said no (top {limit}) --")
    shown = 0
    for prob, sentence, _, is_pos in rows:
        if not is_pos or shown >= limit:
            continue
        has_cue = any(pat.search(sentence) for pat in STRONG.values())
        tag = "cue present" if has_cue else "NO RULE CUE AT ALL"
        print(f"  [{prob:.2f}] ({tag}) {sentence[:110]}")
        shown += 1
    if shown == 0:
        print("  (none)")

    print(f"\n-- rules said action item, model says no (top {limit}) --")
    shown = 0
    for prob, sentence, _, is_pos in reversed(rows):
        if is_pos or shown >= limit:
            continue
        print(f"  [{prob:.2f}] {sentence[:110]}")
        shown += 1
    if shown == 0:
        print("  (none)")
    print()


def hand_labelled_score():
    """Score the model against human labels, if label_tool.py has been run."""
    import csv

    path = ROOT / "test_hand.csv"
    if not path.exists():
        print("=" * 68)
        print("No test_hand.csv -- the numbers above measure agreement with the")
        print("regex labeller, NOT true action-item detection. Run:")
        print("    python label_tool.py 120")
        print("to hand-label a test sample and get an honest number.")
        print("=" * 68 + "\n")
        return

    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)]
    if not rows:
        return

    sentences = [r["sentence"] for r in rows]
    y_true = [int(r["hand_label"]) for r in rows]
    y_rule = [int(r["rule_label"]) for r in rows]

    tokenizer, model = load_model()
    probs = score(tokenizer, model, sentences)
    y_model = [int(p >= 0.5) for p in probs]

    print("=" * 68)
    print(f"AGAINST {len(rows)} HAND LABELS (the honest number)")
    print("=" * 68)
    for name, pred in (("regex labeller", y_rule), ("roberta-base", y_model)):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", pos_label=1, zero_division=0
        )
        print(f"  {name:<18} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
    print(f"\n  hand-labelled positives: {sum(y_true)}/{len(y_true)}")
    print()


def main():
    baseline = load_results("results_baseline.json")
    roberta = load_results("results_roberta.json")
    if not baseline and not roberta:
        raise SystemExit("no results files -- run baseline.py and train.py first")
    table(baseline, roberta)
    if roberta and (ROOT / "action_item_model").exists():
        hand_labelled_score()
        disagreements()


if __name__ == "__main__":
    main()
