"""Weak-label meeting sentences as action item (1) / not (0).

Rules only -- no LLM is involved in producing labels.

The important design choice is HARD NEGATIVES. A sentence that fires a cue but
fails a check ("we have to think about it", "I must say", "we don't have to do
that", "if we had to redesign it") is labelled 0, not dropped. Training only on
cue-free negatives taught the first model that a cue implies an action item,
which wrecked precision on real transcripts. The contrast cases are the lesson.

Usage: python weak_label.py transcripts
Writes: labeled.csv, review.csv (positives), rejected.csv (hard negatives)
"""
import csv
import random
import re
import sys
from pathlib import Path

SEED = 42
MAX_POSITIVES = 500
MAX_HARD_NEG = 650
MAX_EASY_NEG = 650
MIN_WORDS = 4
MAX_WORDS = 45
# a task verb must appear within this many characters after the cue
PROXIMITY = 60

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# --- cues: commitment, directive, assignment, deadline ---------------------
STRONG = {
    "commit": re.compile(r"\b(i|we)\s*(?:'ll|will|shall)\b", re.I),
    "going_to": re.compile(r"\b(?:i'm|i am|we're|we are)\s+(?:going to|gonna)\b", re.I),
    "need_to": re.compile(r"\b(?:we|you|i)\s+(?:need to|have to|must|gotta|hafta)\b", re.I),
    "request": re.compile(r"\b(?:can|could|would|will)\s+you\b|\bplease\b", re.I),
    "assign": re.compile(
        r"\b(?:action item|take care of|takes care of|follow[- ]up on|"
        r"sign off on|circulate (?:the|a|it|that|this)|"
        r"send (?:out|over|me|you|him|her|them)|write up|"
        r"draft (?:the|a|an|up)|schedule a|set up a (?:meeting|call|time)|"
        r"put together|get back to (?:you|me|us|him|her|them)|report back|"
        r"look into (?:the|a|it|that|this)|chase up)\b",
        re.I,
    ),
    "deadline": re.compile(
        r"\bby (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"tomorrow|next week|next month|the end of (?:the )?(?:day|week|month)|"
        r"eod|eow)\b|\bdeadline\b|\bdue (?:date|by|on)\b",
        re.I,
    ),
}

# --- concrete, assignable acts ---------------------------------------------
TASK_VERB = re.compile(
    r"\b(?:send|sends|sending|email|e-mail|mail|write|writes|writing|"
    r"draft|drafts|drafting|prepare|prepares|preparing|produce|producing|"
    r"review|reviews|reviewing|check|checks|checking|verify|verifying|"
    r"update|updates|updating|create|creates|creating|make|makes|making|"
    r"build|builds|building|design|designs|designing|schedule|schedules|"
    r"scheduling|book|books|booking|arrange|arranges|arranging|organise|"
    r"organize|organising|organizing|call|calls|calling|phone|contact|"
    r"contacts|contacting|confirm|confirms|confirming|finalise|finalize|"
    r"finalising|finalizing|circulate|circulates|circulating|share|shares|"
    r"sharing|distribute|distributing|submit|submits|submitting|deliver|"
    r"delivers|delivering|fix|fixes|fixing|test|tests|testing|add|adds|"
    r"adding|remove|removes|removing|delete|order|orders|ordering|purchase|"
    r"buy|install|installs|installing|sign|signs|signing|approve|approves|"
    r"publish|publishes|publishing|document|documents|measure|measures|"
    r"measuring|collect|collects|collecting|gather|gathers|gathering|"
    r"analyse|analyze|analysing|analyzing|calculate|calculates|estimate|"
    r"estimates|print|prints|printing|upload|uploads|download|forward|"
    r"forwards|copy|copies|bring|brings|bringing|chase|chases|remind|"
    r"reminds|allocate|allocates|assign|assigns|assigning|implement|"
    r"implements|implementing|investigate|investigates|provide|provides|"
    r"providing|supply|supplies|open|opens|opening|hire|hires|hiring|"
    r"pay|pays|paying|ask|asks|asking|put together|set up|sets up|"
    r"setting up|look into|looks into|looking into|follow up|follows up|"
    r"following up|take care of|takes care of|get back to|report back|"
    r"reports back|sort out|sorts out|deal with|deals with|draw up|"
    r"draws up|sign off|signs off|write up|writes up|come back to|"
    r"let (?:me|us|you) know|send (?:out|over|round)|take minutes|"
    r"make minutes|present|presents|presenting|"
    # added after reviewing rejected.csv: these were losing real action items
    r"post|posts|posting|close|closes|closing|develop|develops|developing|"
    r"evaluate|evaluates|evaluation|finish|finishes|finishing|complete|"
    r"completes|show|shows|showing|amend|amends|revise|revises|rewrite|"
    r"record|records|recording|replace|replaces|repair|clean|draw|draws|"
    r"sketch|sketches|plan|plans|planning|budget|cost|costs|quote|"
    r"negotiate|ship|ships|run|runs|running|monitor|monitors|track|"
    r"tracks|notify|notifies|inform|informs|invite|invites|attend|"
    r"attends|join|joins|escalate|escalates|flag|flags|raise|raises|"
    r"fill in|fills in|hand in|hands in|hand out|hands out|pass on|"
    r"find out|finds out|make sure|makes sure|put up|puts up|set out|"
    r"lay out|work on|works on|work out|works out|talk to|talks to|"
    r"speak to|speaks to|meet with|meets with|type up|note down)\b",
    re.I,
)

# --- reject checks: each one turns a cue hit into a HARD NEGATIVE ----------
IDIOM = re.compile(
    r"\b(?:have to say|had to say|must say|have to admit|gotta say|"
    r"needless to say|i'll be honest|we'll see|i'll say|i have to agree|"
    r"have to confess|i'll tell you|i'll bet|mind you)\b",
    re.I,
)
NEGATED = re.compile(
    r"\b(?:do not|don't|does not|doesn't|did not|didn't|will not|won't|"
    r"would not|wouldn't|no)\s+(?:\w+\s+){0,2}?"
    r"(?:have to|need to|has to|must|going to|gonna)\b"
    r"|\bno need to\b|\bnothing (?:to|we) \b",
    re.I,
)
PAST = re.compile(
    r"\b(?:had to|needed to|was going to|were going to|i told|we told|"
    r"was supposed to|were supposed to|has been|have been)\b", re.I
)
HYPOTHETICAL = re.compile(
    r"^\s*(?:if|suppose|supposing|imagine)\b"
    r"|\b(?:if we|if you|if i|if they|if it|in case|what if|unless|"
    r"would have to|might have to|could have to|would need to|"
    r"might need to|may need to|would be|rather than|instead of)\b",
    re.I,
)
# A cue followed by a mental or discourse verb is deliberation, not a task.
# Generic auxiliaries (do/be/have/get/go/take/want) are deliberately NOT here:
# they are already caught by the no_task_verb check, and listing them here
# discarded real items like "we have to do the project evaluation".
EPISTEMIC = re.compile(
    r"^(?:\W+)?(?:just |also |really |still |first |then |now |uh |um )*"
    r"(?:think|thinks|thinking|know|knows|say|says|remember|admit|agree|"
    r"see|feel|believe|understand|consider|imagine|assume|suppose|realise|"
    r"realize|mean|guess|hope|wonder|wait|look at|bear in mind|"
    r"be careful|be aware|talk about|discuss|go into|"
    r"start with|move on|come back|go back)\b",
    re.I,
)
RHETORICAL = re.compile(
    r"\b(?:can you imagine|would you say|can you tell me a bit|"
    r"how would you|what would you|why would you|do we have to|"
    r"why do we have to|can you believe)\b",
    re.I,
)


def split_utterances(path):
    """Yield (speaker, sentence) from a 'Speaker: text' transcript."""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        speaker, _, content = raw.partition(":")
        if not content:
            speaker, content = "", raw
        for sent in SENT_SPLIT.split(content.strip()):
            sent = sent.strip()
            if sent:
                yield speaker.strip(), sent


def classify(sentence):
    """Return (label, rules, reason).

    label 1 = action item, 0 = not. reason explains a hard negative, or is ''
    for a positive / a sentence with no cue at all.
    """
    hits = {name: m for name, m in
            ((n, p.search(sentence)) for n, p in STRONG.items()) if m}
    if not hits:
        return 0, "", "no_cue"

    rules = "|".join(sorted(hits))

    for name, pattern in (
        ("idiom", IDIOM),
        ("rhetorical", RHETORICAL),
        ("negated", NEGATED),
        ("hypothetical", HYPOTHETICAL),
        ("past", PAST),
    ):
        if pattern.search(sentence):
            return 0, rules, name

    # an 'assign' cue already names a concrete act
    if "assign" in hits:
        return 1, rules, ""

    cue_end = min(m.end() for name, m in hits.items() if name != "deadline") \
        if any(n != "deadline" for n in hits) else 0

    # what directly follows the cue decides it
    tail = sentence[cue_end:cue_end + PROXIMITY]
    if EPISTEMIC.match(tail.strip()):
        return 0, rules, "epistemic"

    if TASK_VERB.search(tail):
        if sentence.rstrip().endswith("?") and "request" not in hits:
            return 0, rules, "question"
        return 1, rules, ""

    return 0, rules, "no_task_verb"


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "transcripts")
    files = sorted(folder.glob("*.txt"))
    if not files:
        sys.exit(f"no .txt files in {folder}")

    positives, hard_negatives, easy_negatives = [], [], []
    reasons = {}
    for path in files:
        for speaker, sentence in split_utterances(path):
            n_words = len(sentence.split())
            if n_words < MIN_WORDS or n_words > MAX_WORDS:
                continue
            label, rules, reason = classify(sentence)
            row = {
                "source": path.name,
                "speaker": speaker,
                "sentence": sentence,
                "label": label,
                "rules": rules,
                "reason": reason,
            }
            if label == 1:
                positives.append(row)
            elif reason == "no_cue":
                easy_negatives.append(row)
            else:
                hard_negatives.append(row)
                reasons[reason] = reasons.get(reason, 0) + 1

    rng = random.Random(SEED)
    for bucket in (positives, hard_negatives, easy_negatives):
        rng.shuffle(bucket)

    rows = (
        positives[:MAX_POSITIVES]
        + hard_negatives[:MAX_HARD_NEG]
        + easy_negatives[:MAX_EASY_NEG]
    )
    rng.shuffle(rows)

    fields = ["source", "speaker", "sentence", "label", "rules", "reason"]
    for name, subset in (
        ("labeled.csv", rows),
        ("review.csv", [r for r in rows if r["label"] == 1]),
        ("rejected.csv", [r for r in rows if r["label"] == 0 and r["reason"] != "no_cue"]),
    ):
        with open(name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)

    n_pos = sum(r["label"] for r in rows)
    print(f"files:          {len(files)}")
    print(f"positives:      {len(positives)} (kept {min(len(positives), MAX_POSITIVES)})")
    print(f"hard negatives: {len(hard_negatives)} (kept {min(len(hard_negatives), MAX_HARD_NEG)})")
    print(f"easy negatives: {len(easy_negatives)} (kept {min(len(easy_negatives), MAX_EASY_NEG)})")
    print(f"labeled.csv:    {len(rows)} rows, {n_pos} positives "
          f"({n_pos / len(rows):.1%})")
    print("\nhard-negative reasons:")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<14} {count}")
    print("\npositive rule hits:")
    for name in STRONG:
        n = sum(1 for r in rows if r["label"] == 1 and name in r["rules"].split("|"))
        print(f"  {name:<10} {n}")


if __name__ == "__main__":
    main()
