"""TF-IDF + LogisticRegression baseline. Fallback model and comparison number.

Usage: python baseline.py
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from data import SEED, load_split

OUT = Path(__file__).parent / "results_baseline.json"


def main():
    train_df, test_df = load_split()
    print(f"train {len(train_df)} rows ({train_df.label.sum()} pos)")
    print(f"test  {len(test_df)} rows ({test_df.label.sum()} pos)\n")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vec.fit_transform(train_df.sentence)
    x_test = vec.transform(test_df.sentence)

    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED
    )
    clf.fit(x_train, train_df.label)
    pred = clf.predict(x_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df.label, pred, average="binary", pos_label=1, zero_division=0
    )
    print(classification_report(test_df.label, pred, digits=3, zero_division=0))
    print("confusion matrix [[tn fp] [fn tp]]:")
    print(confusion_matrix(test_df.label, pred))
    print(f"\nACTION-ITEM CLASS  precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

    OUT.write_text(
        json.dumps(
            {
                "model": "tfidf+logreg",
                "features": f"{x_train.shape[1]} ngrams",
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "test_rows": int(len(test_df)),
                "test_positives": int(test_df.label.sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
