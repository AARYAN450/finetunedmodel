# Action-Item Extractor (fine-tuned RoBERTa)

Takes a meeting transcript, returns action items as JSON: `task`, `owner`, `due`,
`confidence`.

**No LLM API is called anywhere** — not at inference time, and not to produce
labels. The classifier is `roberta-base` (pretrained weights) fine-tuned by us on
rule-labelled meeting sentences. Owner and due-date extraction is regex-based.

## Approach

1. **Data** — 60 real meetings from [QMSum](https://github.com/Yale-LILY/QMSum)
   (AMI product-design, ICSI academic, and Welsh parliamentary committee
   transcripts). Human-transcribed; no generated text. 3 further meetings are
   held out in `heldout/` and never labelled or trained on.
2. **Weak labelling** (`weak_label.py`) — regex cues for commitment (`I'll`,
   `we need to`), directives (`can you`, `please`), assignment (`action item`,
   `follow up on`) and deadlines. A cue only counts if a **concrete task verb
   follows it within 60 characters**, which separates "we'll send the report"
   from "we'll go into that later". Crucially, a sentence that fires a cue but
   fails a check becomes a **hard negative** (label 0), not a dropped row — see
   Results for why this matters more than anything else here. Reject reasons are
   recorded in `rejected.csv`.
3. **Baseline** (`baseline.py`) — TF-IDF (1–2 grams) + LogisticRegression with
   `class_weight="balanced"`.
4. **Fine-tune** (`train.py`) — `roberta-base`, `max_length=64`, batch 16,
   3 epochs, lr 2e-5, class-weighted cross-entropy. Plain PyTorch loop.
5. **Extraction** (`extract.py`) — classify every sentence, then attach an owner
   (speaker for `I'll…`, named subject for `Dave will…`, `team` for `we'll…`) and
   a due date (weekday/relative-date regexes).

Both `baseline.py` and `train.py` import the split from `data.py`, so the
comparison is on identical rows (seed 42, stratified, 20% test).

## Results

Positive class = action item. Identical split for both models (seed 42,
stratified, 20% test). See `results_baseline.json` / `results_roberta.json`.

| model | precision | recall | F1 |
|---|---|---|---|
| TF-IDF + LogisticRegression (baseline) | 0.554 | 0.671 | **0.607** |
| `roberta-base` fine-tuned, threshold 0.50 | 0.550 | 0.789 | **0.649** |
| `roberta-base`, tuned threshold 0.80 | 0.625 | 0.724 | **0.671** |

Trained on CPU in 615s (3 epochs, batch 16, max_len 64, lr 2e-5).
Test set: 336 sentences, 76 action items.

### Why these numbers are lower than the first run, and better

The first version of the labeller scored F1 **0.895** against a baseline of
0.849. Those numbers were inflated and the model built on them was close to
useless on a real transcript, because every negative example was cue-free. The
model only ever saw "we'll send the report" (positive) against "the weather is
nice" (negative), so it learned *cue present -> action item* and fired on 7-12%
of every transcript.

The fix was **hard negatives**: sentences that fire a cue but fail a check are
now labelled 0 instead of being dropped — idioms (`I must say`), negation
(`we don't have to`), hypotheticals (`if we had to`), mental verbs
(`we need to think about it`) and cues with no concrete task verb. The dataset
carries 650 of them.

That makes the task genuinely harder, and the scores honest:

| | v1 (dropped ambiguous) | v2 (hard negatives) |
|---|---|---|
| baseline F1 | 0.849 | 0.607 |
| RoBERTa F1 | 0.895 | 0.671 |
| agreement with the rules | 95.1% | 80.7% |
| items flagged on `meeting_060` | 43 / 370 | 15 / 370 (thr 0.8) |
| usable items, by inspection | ~1 in 4 | ~1 in 2 |

The baseline collapsing from 0.849 to 0.607 is the point: hard negatives share
vocabulary with positives, so bag-of-words cannot separate them. The gap
between 0.607 and 0.671 is the part that needs context rather than keywords,
which is the whole argument for fine-tuning a transformer here.

### Did it just learn the keyword rules?

Less than before. Agreement with the labeller fell from 95.1% to 80.7%, and the
disagreements favour the model. It finds real items the rules rejected:

- "Um, we have to decide on a target group and the functions of the remote control."
- "So anyway, I'll be putting together stuff."
- "I'll talk about the new project requirements I've just received"

and it rejects rule false positives, having learned that `I must say` and
`I have to say` are idioms rather than commitments. It is correcting its own
training labels in both directions.

## How to run

```bash
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe prepare_transcripts.py rawdata/train.jsonl
.venv/Scripts/python.exe weak_label.py transcripts
.venv/Scripts/python.exe baseline.py
.venv/Scripts/python.exe train.py
.venv/Scripts/python.exe evaluate.py
.venv/Scripts/python.exe extract.py heldout/meeting_060.txt --threshold 0.6
.venv/Scripts/python.exe write_report.py      # results table -> README

# optional but recommended: hand-label a test sample for an honest number
.venv/Scripts/python.exe label_tool.py 120
```

Example output is committed as `items_meeting_060.json` (43 items from 370
sentences). Each item is `task`, `owner`, `due`, `confidence`, plus `said_by`.

## Limitations

Read these before trusting the headline number.

- **The test labels come from the same rules the model was trained on.** The
  reported F1 measures agreement with the regex labeller on held-out sentences,
  not true action-item detection. A model that perfectly learned the regexes
  would score ~1.0. `evaluate.py` prints the model/rule disagreements to show
  where it departs from them; for a trustworthy number, hand-label a test
  sample and re-score.
- **Weak-label bias.** Action items phrased without a cue word ("Owner: the
  slides, by Thursday") are invisible to the labeller, so they are
  under-represented in training and the recall estimate is optimistic.
- **Sentence-level split, not meeting-level.** Sentences from one meeting appear
  in both train and test, so topic and speaker overlap inflate scores somewhat.
- **Owner and due are rules, not learned.** Second-person requests with no
  vocative ("can you send that over?") yield `owner: null`. Relative dates are
  returned as written ("by Friday"), not resolved to a calendar date.
- **Small data.** ~1.4k sentences, ~325 positives, ~285 test rows. Differences
  under a few F1 points are noise at this size.
- **Domain.** Mostly committee and product-design meetings; wording differs from
  typical corporate stand-ups.
- **Whole-transcript precision is still the weak point: roughly half.** At
  threshold 0.8 the model flags 15/370, 27/558 and 15/218 sentences on the three
  held-out meetings, down from 43/370 in v1. By inspection about half are action
  items a person would actually write down. The rest are capability or
  suggestion statements the classifier cannot separate from commitments at this
  data size — "We can put the logo on our base station", "It has to be
  developed". `extract.py` applies a rule-based non-commitment filter after
  scoring (disable with `--no-filter`) which removes the clearest cases;
  it dropped 6 of 27 items on `meeting_061`. Treat the output as a shortlist to
  triage, not a finished action plan. The
  test set is 23% positive by construction; a real transcript is a few percent,
  so the same precision hurts far more. Raising `--threshold` barely helps,
  because the model is badly overconfident — most positives sit above 0.99.
  Calibration, or a precision-oriented threshold chosen on hand-labelled data,
  is the obvious next step.
- **Nothing here is hand-verified.** The plan forbids LLM-generated labels, and
  that includes labels written by the assistant that built this, so no manual
  label pass was performed. `label_tool.py` exists for a human to do it; until
  someone runs it, every number in this README is agreement with regexes.
