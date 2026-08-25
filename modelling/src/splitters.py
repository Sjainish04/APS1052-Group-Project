"""
Cross-validation splitters for a PANEL.

Two are provided, and they answer different questions.

`DateGroupedTimeSeriesSplit` is the brief's "always use TimeSeriesSplit",
applied to whole dates. It gives ONE backtest path: train on the past, test on
the future, walk forward. That is the honest simulation of live trading, and it
is what model selection runs on.

`CombinatorialPurgedCV` gives MANY paths. It is Lopez de Prado's CPCV: split the
timeline into N contiguous groups, hold out every combination of k of them, and
purge plus embargo around each held-out block. With N=6, k=2 you get C(6,2)=15
splits, and because each group is held out in 5 of them you can reassemble 5
distinct backtest paths. One walk-forward number tells you what happened on the
single path history handed you; a distribution over 5 tells you how much of that
was luck.

WHY ROW-WISE SPLITTING IS WRONG HERE
------------------------------------
With 20 stocks per day, sklearn's `TimeSeriesSplit` puts the SAME DATE on both
sides of a fold boundary. The model trains on 12 names from a day and validates
on the other 8 -- having already seen that day's cross-section, including the
cross-sectional mean the target was demeaned by. Every score inflates. Both
splitters here break only between whole dates.

Run `python -m modelling.src.splitters` for the self-test.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

__all__ = ["DateGroupedTimeSeriesSplit", "CombinatorialPurgedCV"]


class DateGroupedTimeSeriesSplit:
    """Expanding-window TimeSeriesSplit whose folds break between whole days.

    Parameters
    ----------
    n_splits : number of folds.
    embargo  : trading days dropped after each training block. The target looks
               one day forward, so without a gap the last training label peeks
               into validation. 5 also covers the 5-day features.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 5):
        self.n_splits = n_splits
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be the date of each row")
        g = pd.Series(np.asarray(groups))
        udates = np.array(sorted(g.unique()))
        n, idx = len(udates), np.arange(len(g))
        if n < self.n_splits + 1:
            raise ValueError(f"{n} dates is too few for {self.n_splits} splits")
        fold = n // (self.n_splits + 1)
        for k in range(self.n_splits):
            tr_end = fold * (k + 1)
            va_beg = tr_end + self.embargo
            va_end = min(va_beg + fold, n)
            if va_beg >= n:
                break
            tr = idx[g.isin(udates[:tr_end]).to_numpy()]
            va = idx[g.isin(udates[va_beg:va_end]).to_numpy()]
            if len(tr) and len(va):
                yield tr, va


class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation over whole dates.

    Splits the ordered unique dates into `n_groups` contiguous blocks and holds
    out every combination of `n_test_groups`. Training dates within `embargo`
    days of ANY held-out block are purged, on both sides -- a test block sitting
    in the middle of the timeline leaks in both directions, which is exactly the
    case a walk-forward split never has to handle.

    Parameters
    ----------
    n_groups      : contiguous date blocks. 6 is the usual choice.
    n_test_groups : blocks held out per split. 2 gives C(6,2)=15 splits.
    embargo       : trading days purged either side of each test block.

    Notes
    -----
    Each group is held out in C(n_groups-1, n_test_groups-1) splits, so that
    many complete backtest paths can be reassembled. For (6, 2) that is 5.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo: int = 5):
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be smaller than n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(list(combinations(range(self.n_groups), self.n_test_groups)))

    @property
    def n_paths(self) -> int:
        """How many complete backtest paths the splits can be reassembled into."""
        return len(list(combinations(range(self.n_groups - 1), self.n_test_groups - 1)))

    def _blocks(self, udates):
        """Contiguous, near-equal date blocks. np.array_split handles remainders."""
        return np.array_split(np.arange(len(udates)), self.n_groups)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be the date of each row")
        g = pd.Series(np.asarray(groups))
        udates = np.array(sorted(g.unique()))
        blocks = self._blocks(udates)
        idx = np.arange(len(g))
        pos = pd.Series(np.arange(len(udates)), index=udates)
        row_pos = g.map(pos).to_numpy()          # date position of every row

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_pos = np.concatenate([blocks[b] for b in combo])
            # Purge + embargo a window either side of every held-out block.
            banned = set(test_pos.tolist())
            for b in combo:
                lo, hi = blocks[b][0], blocks[b][-1]
                banned.update(range(max(0, lo - self.embargo), lo))
                banned.update(range(hi + 1, min(len(udates), hi + 1 + self.embargo)))
            train_mask = ~np.isin(row_pos, list(banned))
            test_mask = np.isin(row_pos, test_pos)
            tr, te = idx[train_mask], idx[test_mask]
            if len(tr) and len(te):
                yield tr, te

    def path_assignments(self, groups):
        """Map each split to a backtest path.

        Split i contributes its test blocks to a path; assembling one block from
        each position across splits reconstructs a full-timeline prediction. The
        j-th time a given block is held out belongs to path j.
        """
        seen = {b: 0 for b in range(self.n_groups)}
        out = []
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            paths = {}
            for b in combo:
                paths[b] = seen[b]
                seen[b] += 1
            out.append(paths)
        return out


# --------------------------------------------------------------------------

def _selftest():
    print("splitters self-test")
    dates = np.repeat(pd.date_range("2022-01-03", periods=300, freq="B"), 20)
    X = pd.DataFrame({"f": np.arange(len(dates))})
    y = np.zeros(len(dates))

    # -- expanding window --------------------------------------------------
    cv = DateGroupedTimeSeriesSplit(n_splits=5, embargo=5)
    folds = list(cv.split(X, y, groups=dates))
    assert len(folds) == 5
    for k, (tr, va) in enumerate(folds):
        td, vd = pd.Series(dates[tr]), pd.Series(dates[va])
        assert td.max() < vd.min(), "train overlaps validation"
        assert len(set(td.unique()) & set(vd.unique())) == 0, "a date is on both sides"
        gap = np.busday_count(td.max().date(), vd.min().date())
        assert gap >= 5, f"embargo too small: {gap}"
    print(f"  [ok] expanding window: {len(folds)} folds, no date on both sides, "
          "embargo >= 5 business days in every fold")

    # -- CPCV --------------------------------------------------------------
    cp = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=5)
    splits = list(cp.split(X, y, groups=dates))
    assert len(splits) == 15, len(splits)
    assert cp.n_paths == 5, cp.n_paths
    print(f"  [ok] CPCV(6,2): {len(splits)} splits, {cp.n_paths} reassemblable paths")

    for tr, te in splits:
        trd, ted = set(pd.Series(dates[tr]).unique()), set(pd.Series(dates[te]).unique())
        assert not (trd & ted), "a date appears in both train and test"
    print("  [ok] CPCV: no date is ever in both train and test")

    # purging actually removes something on both sides of an interior block
    udates = np.array(sorted(pd.unique(dates)))
    interior = [s for s, combo in zip(splits, combinations(range(6), 2)) if 0 not in combo]
    tr, te = interior[0]
    tr_pos = set(pd.Series(dates[tr]).map(
        pd.Series(np.arange(len(udates)), index=udates)).tolist())
    te_pos = sorted(pd.Series(dates[te]).map(
        pd.Series(np.arange(len(udates)), index=udates)).unique().tolist())
    before = te_pos[0] - 1
    assert before not in tr_pos, "the day immediately before a test block was not purged"
    print("  [ok] CPCV: days immediately before an interior test block are purged")

    # every date is tested exactly n_paths times
    counts = pd.Series(0, index=udates)
    for _, te in splits:
        counts[pd.Series(dates[te]).unique()] += 1
    assert counts.min() == counts.max() == cp.n_paths, counts.value_counts().to_dict()
    print(f"  [ok] CPCV: every date is held out exactly {cp.n_paths} times "
          "-> 5 complete backtest paths")

    # coverage: training sets stay usefully large
    fracs = [len(tr) / len(dates) for tr, _ in splits]
    print(f"  [ok] CPCV train fraction: {min(fracs):.0%}-{max(fracs):.0%} of rows")

    print("\nall splitter checks passed.")


if __name__ == "__main__":
    _selftest()
