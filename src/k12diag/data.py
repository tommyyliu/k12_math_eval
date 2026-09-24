"""Loading the Eedi NeurIPS 2020 Task 3/4 data and building the evaluation split."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "data"


def load_task34(raw: Path = RAW) -> pd.DataFrame:
    """One row per (student, question) answer, with dense 0-based ids and answer dates."""
    df = pd.read_csv(raw / "train_data" / "train_task_3_4.csv")
    meta = pd.read_csv(
        raw / "metadata" / "answer_metadata_task_3_4.csv",
        usecols=["AnswerId", "DateAnswered", "QuizId"],
    )
    df = df.merge(meta, on="AnswerId", how="left")
    df["DateAnswered"] = pd.to_datetime(df["DateAnswered"])
    df["student"] = df["UserId"].astype("category").cat.codes.astype(np.int64)
    df["item"] = df["QuestionId"].astype("category").cat.codes.astype(np.int64)
    return df


def load_question_subjects(raw: Path = RAW) -> pd.DataFrame:
    """QuestionId -> most specific subject name (deepest level in the topic tree)."""
    qm = pd.read_csv(raw / "metadata" / "question_metadata_task_3_4.csv")
    sm = pd.read_csv(raw / "metadata" / "subject_metadata.csv").set_index("SubjectId")
    rows = []
    for qid, subj in zip(qm["QuestionId"], qm["SubjectId"]):
        ids = [int(s) for s in subj.strip("[]").split(",")]
        ids = [s for s in ids if s in sm.index]
        deepest = max(ids, key=lambda s: sm.loc[s, "Level"])
        level2 = [s for s in ids if sm.loc[s, "Level"] == 2]
        rows.append(
            {
                "QuestionId": qid,
                "subject": sm.loc[deepest, "Name"],
                "area": sm.loc[level2[0], "Name"] if level2 else None,
            }
        )
    return pd.DataFrame(rows)


def correct_options(df: pd.DataFrame) -> np.ndarray:
    """0-based index of the correct option for every item."""
    return df.groupby("item")["CorrectAnswer"].first().sort_index().to_numpy() - 1


def densest_window(dates: pd.Series, days: int) -> pd.Series:
    """Boolean mask selecting the answers in the `days`-long window containing the most answers."""
    t = dates.sort_values()
    ns = t.values.astype("datetime64[ns]").astype(np.int64)
    width = np.int64(days) * 86_400 * 10**9
    ends = np.searchsorted(ns, ns + width, side="left")
    start = int(np.argmax(ends - np.arange(len(ns))))
    lo, hi = ns[start], ns[start] + width
    v = dates.values.astype("datetime64[ns]").astype(np.int64)
    return pd.Series((v >= lo) & (v < hi), index=dates.index)


@dataclass
class TestStudent:
    student: int
    pool_items: np.ndarray
    pool_y: np.ndarray
    target_items: np.ndarray
    target_y: np.ndarray
    pool_opt: np.ndarray | None = None  # chosen option, 0-based
    target_opt: np.ndarray | None = None


@dataclass
class Split:
    train: pd.DataFrame
    test: list[TestStudent]
    n_students: int
    n_items: int
    bank: np.ndarray  # items with enough training responses to estimate; the only ones asked or scored


def make_split(
    df: pd.DataFrame,
    n_test: int = 1000,
    window_days: int = 90,
    min_window_answers: int = 80,
    target_frac: float = 0.3,
    min_item_train: int = 100,
    seed: int = 0,
) -> Split:
    """Hold out whole students. For each, keep the densest time window and split it into pool/target.

    Items with fewer than `min_item_train` training responses are dropped from the bank:
    their parameters are too noisy, and the selection policies otherwise favor them.
    """
    rng = np.random.default_rng(seed)
    in_window = df.groupby("student", group_keys=False)["DateAnswered"].apply(
        lambda s: densest_window(s, window_days)
    )
    df = df.assign(in_window=in_window.reindex(df.index))
    counts = df[df["in_window"]].groupby("student").size()
    eligible = counts.index[counts >= min_window_answers].to_numpy()
    test_ids = rng.choice(eligible, size=min(n_test, len(eligible)), replace=False)

    train = df[~df["student"].isin(test_ids)]
    item_counts = train.groupby("item").size()
    bank = np.sort(item_counts.index[item_counts >= min_item_train].to_numpy())

    test = []
    windowed = df[df["in_window"] & df["student"].isin(test_ids) & df["item"].isin(bank)]
    for sid, g in windowed.groupby("student"):
        perm = rng.permutation(len(g))
        n_target = int(round(target_frac * len(g)))
        items, y = g["item"].to_numpy(), g["IsCorrect"].to_numpy()
        opt = g["AnswerValue"].to_numpy() - 1
        t, p = perm[:n_target], perm[n_target:]
        test.append(TestStudent(int(sid), items[p], y[p], items[t], y[t], opt[p], opt[t]))

    return Split(
        train=train,
        test=test,
        n_students=int(df["student"].max()) + 1,
        n_items=int(df["item"].max()) + 1,
        bank=bank,
    )
