"""QMSum jsonl -> plain-text transcripts.

Source: Yale-LILY/QMSum (AMI product-design, ICSI academic, parliamentary
committee meetings). Human transcripts, no LLM involvement.

Usage: python prepare_transcripts.py rawdata/train.jsonl
"""
import json
import re
import sys
from pathlib import Path

N_TRAIN_MEETINGS = 60
N_HELDOUT_MEETINGS = 3

# AMI/ICSI annotation markers: {vocalsound} {disfmarker} {gap} {pause} ...
MARKER = re.compile(r"\{[^}]*\}")
# transcripts put a space before punctuation: "Okay ." -> "Okay."
SPACED_PUNCT = re.compile(r"\s+([.,!?;:])")
SPACED_APOS = re.compile(r"\s+'")
WS = re.compile(r"\s+")


def clean(text):
    text = MARKER.sub(" ", text)
    text = SPACED_PUNCT.sub(r"\1", text)
    text = SPACED_APOS.sub("'", text)
    text = WS.sub(" ", text)
    return text.strip()


def write_meeting(path, utterances):
    lines = []
    for utt in utterances:
        content = clean(utt["content"])
        if not content:
            continue
        speaker = WS.sub(" ", utt["speaker"]).strip()
        lines.append(f"{speaker}: {content}")
    if not lines:
        return False
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "rawdata/train.jsonl")
    root = Path(__file__).parent
    train_dir = root / "transcripts"
    heldout_dir = root / "heldout"
    train_dir.mkdir(exist_ok=True)
    heldout_dir.mkdir(exist_ok=True)

    with src.open(encoding="utf-8") as fh:
        meetings = [json.loads(line) for line in fh if line.strip()]

    total = N_TRAIN_MEETINGS + N_HELDOUT_MEETINGS
    written = 0
    for i, meeting in enumerate(meetings[:total]):
        out_dir = train_dir if i < N_TRAIN_MEETINGS else heldout_dir
        name = f"meeting_{i:03d}.txt"
        if write_meeting(out_dir / name, meeting["meeting_transcripts"]):
            written += 1

    print(f"meetings read:    {len(meetings)}")
    print(f"transcripts/:     {len(list(train_dir.glob('*.txt')))} files")
    print(f"heldout/:         {len(list(heldout_dir.glob('*.txt')))} files")
    print(f"written:          {written}")


if __name__ == "__main__":
    main()
