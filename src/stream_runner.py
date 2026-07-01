"""Stream runner: baseframework integration plus the anytime stream workflow.

All baseframework access is here so we reuse its data loading and metrics rather
than reimplementing them: load_dataset -> datahandler.load_data, compute_cmm ->
metrics/cmm, compute_temporal_silhouette -> metrics/TSindex. StreamRunner
feeds the dataset as an anytime stream.
"""
from __future__ import annotations

import contextlib
import os
import signal
import threading
import time

import numpy as np

import utils
from . import BASEFRAMEWORK_DIR, setup_baseframework_path
from .microcluster import MicroCluster


@contextlib.contextmanager
def _in_baseframework_dir():
    # datahandler reads relative ./data paths, so run it from the submodule dir.
    prev = os.getcwd()
    os.chdir(BASEFRAMEWORK_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


def load_dataset(name, batch=None, batch_max=None, scale=True):
    """Load a baseframework dataset (complex9 / rbf3 / har / stream_densired)."""
    setup_baseframework_path()
    import datahandler
    with _in_baseframework_dir():
        return datahandler.load_data(name, batch=batch, batch_max=batch_max,
                                     scale=scale)


def compute_cmm(X, gt, clu, weights=None, k=10):
    setup_baseframework_path()
    import cmm
    if weights is None:
        weights = np.ones(len(X))
    return cmm.get_cmm(np.asarray(X), np.asarray(gt), np.asarray(clu),
                       np.asarray(weights), k)


def compute_temporal_silhouette(t, X, labels, s=200, kn=200, c=1):
    setup_baseframework_path()
    import TSindex
    _, _, ts = TSindex.tempsil(np.asarray(t), np.asarray(X), np.asarray(labels),
                               s=s, kn=kn, c=c)
    return ts


class StreamRunner:
    """Feed a dataset into a ClusTree as an anytime stream.

    online sets the speed through its per-object distance budget. stop_after
    stops processing after that many seconds (-1 = all) and pps optionally paces
    the stream in real time (0 = as fast as possible). Ctrl+C also stops the
    stream. On any stop the offline phase runs and the metrics are reported.

    With aggregate=True the Sec. 3.3 speed-up runs: an aggregation thread groups
    similar incoming objects while this (insertion) thread inserts whole
    aggregates full-depth, exactly the two-thread setup from the paper.
    """

    def __init__(self, online, offline=None, stop_after=-1.0, pps=0.0,
                 compute_cmm=True, compute_ts=True, progress=True,
                 aggregate=False, max_aggregates=20):
        self.online = online
        self.offline = offline
        self.stop_after = stop_after
        self.pps = pps
        self.compute_cmm = compute_cmm
        self.compute_ts = compute_ts
        self.progress = progress
        self.aggregate = aggregate
        self.max_aggregates = max_aggregates

    def run(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = None if y is None else np.asarray(y)

        # First Ctrl+C only sets a flag so the stream stops cleanly. During the
        # offline phase below we ignore Ctrl+C completely, so a late or repeated
        # press never kills the process before the results are out.
        self._stop = False
        self._arrived = 0
        in_main = threading.current_thread() is threading.main_thread()
        prev_handler = signal.getsignal(signal.SIGINT) if in_main else None
        if in_main:
            signal.signal(signal.SIGINT, lambda s, f: setattr(self, "_stop", True))

        print("[Ctrl+C stops the stream and runs the offline phase]")
        try:
            if self.aggregate:
                n = self._stream_aggregated(X)
            else:
                n = self._stream_plain(X)

            if in_main:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            result = self._evaluate(X[:n], None if y is None else y[:n], n)
            self._report(result, "interrupted" if self._stop else "final")
            return result
        finally:
            if in_main:
                signal.signal(signal.SIGINT, prev_handler)

    def _stream_plain(self, X):
        """One object per learn_one, budget-limited anytime insertion."""
        delay = 1.0 / self.pps if self.pps and self.pps > 0 else 0.0
        report_every = max(1, len(X) // 20)
        start = time.perf_counter()
        n = 0
        try:
            for i in range(len(X)):
                self.online.learn_one(X[i], t=i + 1)
                n = i + 1
                if self.progress and n % report_every == 0:
                    self._progress(n, len(X), start)
                if self._stop:
                    print("\nstopping: interrupted (Ctrl+C)")
                    break
                if self.stop_after > 0 and (time.perf_counter() - start) >= self.stop_after:
                    print("stopping: %.2fs elapsed" % (time.perf_counter() - start))
                    break
                if delay:
                    time.sleep(delay)
        except KeyboardInterrupt:   # fallback when not on the main thread
            self._stop = True
            print("\nstopping: interrupted (Ctrl+C)")
        return n

    def _stream_aggregated(self, X):
        """Two threads (Sec. 3.3): a producer aggregates arriving objects while
        this thread inserts the fullest aggregate full-depth whenever idle."""
        buffer = _AggregateBuffer(self.max_aggregates, self.online.lam)
        report_every = max(1, len(X) // 20)
        start = time.perf_counter()
        halt = threading.Event()
        done = threading.Event()

        def produce():
            delay = 1.0 / self.pps if self.pps and self.pps > 0 else 0.0
            for i in range(len(X)):
                if self._stop or halt.is_set():
                    break
                buffer.add(MicroCluster.from_point(X[i], i + 1))
                self._arrived = i + 1
                if delay:
                    time.sleep(delay)
            done.set()

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()

        last_report = 0
        try:
            while not self._stop:
                if self.stop_after > 0 and (time.perf_counter() - start) >= self.stop_after:
                    print("stopping: %.2fs elapsed" % (time.perf_counter() - start))
                    break
                cf = buffer.pop_fullest()
                if cf is None:
                    if done.is_set():
                        break
                    time.sleep(0.0005)     # idle: wait for the aggregation thread
                    continue
                self.online.insert_cf(cf, t=max(self._arrived, 1))
                buffer.maxradius = self.online.avg_leaf_radius()
                if self.progress and self._arrived - last_report >= report_every:
                    last_report = self._arrived
                    self._progress(self._arrived, len(X), start,
                                   pending=len(buffer))
            if self._stop:
                print("\nstopping: interrupted (Ctrl+C)")
        except KeyboardInterrupt:
            self._stop = True
            print("\nstopping: interrupted (Ctrl+C)")
        finally:
            halt.set()
            producer.join(timeout=1.0)
        return self._arrived

    def _progress(self, n, total, start, pending=None):
        extra = "" if pending is None else "  pending_aggs=%d" % pending
        print("[progress] %d/%d (%d%%)  microclusters=%d%s  elapsed=%.1fs"
              % (n, total, 100 * n // total, len(self.online.get_summary()),
                 extra, time.perf_counter() - start))

    def _evaluate(self, X_seen, y_seen, n):
        summaries = self.online.get_summary()
        result = {"n_points": n, "n_microclusters": len(summaries)}
        if not summaries or self.offline is None:
            return result

        labels_by_id = self.offline.predict_summaries(summaries)
        ids = list(summaries.keys())
        centers = np.array([summaries[i]["position"] for i in ids], dtype=float)
        point_labels = np.array([labels_by_id.get(ids[j], -1)
                                 for j in self._nearest(X_seen, centers)])
        result["n_clusters"] = len({l for l in labels_by_id.values() if l != -1})

        if y_seen is not None:
            if self.compute_ts:
                result["temporal_silhouette"] = _safe(
                    lambda: compute_temporal_silhouette(np.arange(n), X_seen, point_labels))
            if self.compute_cmm:
                result["cmm"] = _safe(
                    lambda: compute_cmm(X_seen, y_seen, point_labels))
        return result

    @staticmethod
    def _nearest(X_seen, centers, step=1000):
        # assign each seen point to its closest current micro-cluster, in chunks
        nearest = np.empty(len(X_seen), dtype=int)
        for s in range(0, len(X_seen), step):
            d = utils.distance_calc(X_seen[s:s + step], centers)
            nearest[s:s + step] = np.argmin(d, axis=1)
        return nearest

    @staticmethod
    def _report(result, when):
        def fmt(v):
            if v is None:
                return "n/a"
            return "%.4f" % v if isinstance(v, float) else str(v)
        line = "  ".join("%s=%s" % (k, fmt(v)) for k, v in result.items())
        print("[%s] %s" % (when, line))


def _safe(fn):
    # a metric can fail on a degenerate clustering; report it as missing
    try:
        return float(fn())
    except Exception:
        return None


class _AggregateBuffer:
    """Thread-safe pool of aggregates shared by the two aggregation threads.

    The producer calls add() for each arriving object; the consumer calls
    pop_fullest() when idle. maxradius is written by the consumer after each
    insertion and read by the producer (Sec. 3.3).
    """

    def __init__(self, max_aggregates, lam):
        self.max_aggregates = max_aggregates
        self.lam = lam
        self.maxradius = 0.0
        self._aggs = []          # list of [cf, age]
        self._counter = 0
        self._lock = threading.Lock()

    def add(self, x_cf):
        with self._lock:
            if self._aggs:
                target = x_cf.mean()
                dists = [utils.distance_calc(target, a[0].mean()) for a in self._aggs]
                j = int(np.argmin(dists))
                # join the closest aggregate within maxradius, or when full
                if dists[j] <= self.maxradius or len(self._aggs) >= self.max_aggregates:
                    self._aggs[j][0].decay(x_cf.timestamp, self.lam)
                    self._aggs[j][0].add(x_cf)
                    return
            self._aggs.append([x_cf.copy(), self._counter])
            self._counter += 1

    def pop_fullest(self):
        # fullest first, ties broken by oldest
        with self._lock:
            if not self._aggs:
                return None
            i = max(range(len(self._aggs)),
                    key=lambda k: (self._aggs[k][0].n, -self._aggs[k][1]))
            cf, _age = self._aggs.pop(i)
            return cf

    def __len__(self):
        with self._lock:
            return len(self._aggs)
