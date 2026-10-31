"""Hand-label a sample of the TEST split, so the headline number is honest.

The reported F1 otherwise measures agreement with the regex labeller. Labelling
a sample by hand gives a number that measures actual action-item detection.
Labels you type here are human labels -- that is the point; nothing is
auto-filled from the rules.

Usage: python label_tool.py [n]          # default 120 sentences
Writes/updates: test_hand.csv  (resumable -- rerun to continue)

Keys:  y = action item   n = not   s = skip/unsure   q = save and quit
An action item = someone commits to, or is asked to do, a concrete task.
Decisions and opinions are NOT action items.
"""
import csv
import sys
from pathlib import Path

from data import load_split

OUT = Path(__file__).parent / "test_hand.csv"
FIELDS = ["sentence", "hand_label", "rule_label"]


def load_done():
    if not OUT.exists():
        return {}
    with OUT.open(encoding="utf-8", newline="") as fh:
        return {r["sentence"]: r for r in csv.DictReader(fh)}


def save(rows):
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    _, test_df = load_split()
    done = load_done()

    # stable order so reruns continue where you left off
    pool = [
        (row.sentence, int(row.label))
        for row in test_df.itertuples()
        if row.sentence not in done
    ]
    todo = pool[: max(0, target - len(done))]
    if not todo:
        print(f"{len(done)} already labelled in {OUT.name}; nothing to do.")
        return

    print(f"{len(done)} labelled so far, {len(todo)} to go.")
    print("y = action item, n = not, s = unsure, q = save and quit\n")

    for i, (sentence, rule_label) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {sentence}")
        while True:
            key = input("  y/n/s/q > ").strip().lower()
            if key in {"y", "n", "s", "q"}:
                break
            print("  please type y, n, s or q")
        if key == "q":
            break
        if key == "s":
            continue
        done[sentence] = {
            "sentence": sentence,
            "hand_label": 1 if key == "y" else 0,
            "rule_label": rule_label,
        }
        save(done)

    save(done)
    labelled = [r for r in done.values()]
    pos = sum(int(r["hand_label"]) for r in labelled)
    agree = sum(
        1 for r in labelled if int(r["hand_label"]) == int(r["rule_label"])
    )
    print(f"\nsaved {len(labelled)} labels to {OUT.name} ({pos} positives)")
    if labelled:
        print(f"rule labeller agreed with you on {agree}/{len(labelled)} "
              f"({agree / len(labelled):.0%}) -- this is the weak-label accuracy")
    print("now run: python evaluate.py   (it picks up test_hand.csv automatically)")


if __name__ == "__main__":
    main()
