"""Transcript -> JSON action items, using the fine-tuned model.

Owner and due date are rule-based (regex + the transcript's own speaker list),
not learned -- see README limitations.

Usage: python extract.py heldout/meeting_060.txt [--threshold 0.6] [--out items.json]
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from weak_label import split_utterances

MODEL_DIR = Path(__file__).parent / "action_item_model"
MAX_LEN = 64

FIRST_PERSON_SINGULAR = re.compile(r"\b(?:i'll|i will|i'm going to|i am going to|i shall|i need to|i have to)\b", re.I)
FIRST_PERSON_PLURAL = re.compile(r"\b(?:we'll|we will|we're going to|we are going to|we need to|we have to|let's)\b", re.I)
SECOND_PERSON = re.compile(r"\b(?:can|could|would|will)\s+you\b|\byou (?:need to|have to|should|must)\b|\bplease\b", re.I)
# "Dave will send it" / "Dave, can you..."
NAMED_SUBJECT = re.compile(r"\b([A-Z][a-z]{2,})\s+(?:will|should|is going to|needs to|has to|can)\b")
VOCATIVE = re.compile(r"\b([A-Z][a-z]{2,})\s*[,:]")
# capitalised discourse markers are not names -- without this, "Okay, you can
# open the presentation?" is attributed to a person called "Okay"
NOT_A_NAME = {
    "okay", "yeah", "yes", "no", "so", "well", "right", "now", "and", "but",
    "um", "uh", "maybe", "also", "then", "actually", "sure", "alright",
    "anyway", "however", "because", "hmm", "oh", "the", "this", "that",
    "these", "those", "there", "here", "what", "when", "where", "why", "how",
    "who", "which", "just", "like", "look", "listen", "sorry", "thanks",
    "first", "second", "finally", "basically", "obviously", "perhaps",
}

# Model false positives cluster on capability and speculation rather than
# commitment ("we can make this", "maybe we could", "we can't"). The classifier
# alone does not separate these reliably at this data size, so filter them.
NON_COMMITMENT = re.compile(
    r"\b(?:can't|cannot|can not|couldn't|could not|won't|wouldn't|"
    r"don't know|do not know|not sure|i doubt)\b"
    r"|^\s*(?:if|maybe|perhaps|suppose)\b"
    r"|\b(?:maybe|perhaps|possibly|probably|might|may)\s+(?:we|you|i|they)\b"
    r"|\b(?:we|you|i|they)\s+(?:could|can)\s+(?:maybe|perhaps|possibly)\b",
    re.I,
)

DUE_PATTERNS = [
    re.compile(r"\bby (?:next )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
    re.compile(r"\b(tomorrow|today|tonight)\b", re.I),
    re.compile(r"\bby the end of (?:the )?(day|week|month|quarter|year)\b", re.I),
    re.compile(r"\b(?:by|before|within) (next week|next month|next meeting|the next meeting)\b", re.I),
    # FILLER: these are speech transcripts -- "in the next uh twenty minutes"
    re.compile(
        r"\b(?:in|within) (?:the next )?(?:uh |um |er )*([\w-]+) "
        r"(?:uh |um )*(minutes?|mins?|hours?|days?|weeks?|months?)\b",
        re.I,
    ),
    re.compile(r"\bbefore (?:the )?(?:next )?(meeting|lunch|break|call)\b", re.I),
    re.compile(r"\b(?:by|on) (\d{1,2}(?:st|nd|rd|th)? (?:of )?(?:january|february|march|april|may|june|july|august|september|october|november|december))\b", re.I),
    re.compile(r"\b(?:by|on) ((?:january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2})\b", re.I),
    re.compile(r"\b(deadline|due date) (?:is|of) ([^.,;]{3,30})", re.I),
    re.compile(r"\b(next week|next month|next meeting|this week|this afternoon|this morning)\b", re.I),
]


def find_due(sentence):
    for pattern in DUE_PATTERNS:
        match = pattern.search(sentence)
        if match:
            return match.group(0).strip()
    return None


def find_owner(sentence, speaker, known_speakers):
    """Best-effort owner. Returns a name, 'team', or None."""
    named = NAMED_SUBJECT.search(sentence)
    if named and named.group(1).lower() not in NOT_A_NAME:
        return named.group(1)

    if SECOND_PERSON.search(sentence):
        vocative = VOCATIVE.search(sentence)
        if vocative and vocative.group(1).lower() not in NOT_A_NAME:
            return vocative.group(1)
        for name in known_speakers:
            if name and name != speaker and re.search(rf"\b{re.escape(name)}\b", sentence):
                return name
        return None  # directed at someone, but we cannot tell who

    if FIRST_PERSON_SINGULAR.search(sentence):
        return speaker or None
    if FIRST_PERSON_PLURAL.search(sentence):
        return "team"
    return speaker or None


def load_model():
    if not MODEL_DIR.exists():
        raise SystemExit(f"{MODEL_DIR} not found -- run train.py first")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def score(tokenizer, model, sentences, batch_size=32):
    probs = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        enc = tokenizer(
            chunk, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
        )
        logits = model(**enc).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return probs


def as_action_plan(payload):
    """Render the items as a readable action plan grouped by owner."""
    items = payload["action_items"]
    lines = [
        f"# Action plan -- {payload['source']}",
        "",
        f"{len(items)} action items from {payload['sentences_scored']} sentences "
        f"(threshold {payload['threshold']})",
        "",
    ]
    by_owner = {}
    for item in items:
        by_owner.setdefault(item["owner"] or "unassigned", []).append(item)

    # named owners first, then the catch-all buckets
    def sort_key(name):
        return (name in ("team", "unassigned"), name)

    for owner in sorted(by_owner, key=sort_key):
        lines.append(f"## {owner}")
        for item in sorted(by_owner[owner], key=lambda d: -d["confidence"]):
            due = f" **(due: {item['due']})**" if item["due"] else ""
            lines.append(f"- {item['task']}{due}  _[{item['confidence']:.2f}]_")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--format", choices=("json", "md"), default="json",
        help="json for machine use, md for a readable action plan",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="keep only the N highest-confidence items",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="skip the non-commitment filter (show raw classifier output)",
    )
    args = parser.parse_args()

    pairs = list(split_utterances(args.transcript))
    pairs = [(sp, s) for sp, s in pairs if 4 <= len(s.split()) <= 45]
    if not pairs:
        raise SystemExit("no usable sentences found")
    known_speakers = {sp for sp, _ in pairs}

    tokenizer, model = load_model()
    probs = score(tokenizer, model, [s for _, s in pairs])

    items, filtered = [], 0
    for (speaker, sentence), prob in zip(pairs, probs):
        if prob < args.threshold:
            continue
        if not args.no_filter and NON_COMMITMENT.search(sentence):
            filtered += 1
            continue
        items.append(
            {
                "task": sentence,
                "owner": find_owner(sentence, speaker, known_speakers),
                "due": find_due(sentence),
                "confidence": round(float(prob), 3),
                "said_by": speaker,
            }
        )
    items.sort(key=lambda d: -d["confidence"])
    if args.top:
        items = items[: args.top]

    payload = {
        "source": args.transcript.name,
        "sentences_scored": len(pairs),
        "threshold": args.threshold,
        "filtered_non_commitments": filtered,
        "action_items": items,
    }
    text = (
        as_action_plan(payload)
        if args.format == "md"
        else json.dumps(payload, indent=2, ensure_ascii=False)
    )
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(items)} action items)")
    else:
        print(text)


if __name__ == "__main__":
    main()
