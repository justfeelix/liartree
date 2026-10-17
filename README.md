# ClusTree and LiarTree

Implementation of the anytime stream-clustering algorithm **ClusTree** and its extension **LiarTree**, integrated with the provided
baseframework for datasets and evaluation. ClusTree maintains a self-adapting
hierarchy of micro-clusters that can be interrupted at any time; LiarTree builds
on it to handle noise, emerging concepts, and overlapping clusters.

## Project Structure

- `src/microcluster.py`: the cluster feature / micro-cluster `(n, LS, SS)` with
  time decay, and mean, variance, radius and the additive operations both trees
  build on.
- `src/node.py`: the tree building blocks, entries and nodes (including the
  noise-buffer and liar-node slots used by LiarTree).
- `src/clustree.py`: the ClusTree online phase, anytime insertion (buffer and
  hitchhiker), decay, split / merge / discard, the four descent strategies, and
  the fast-stream aggregation hook.
- `src/liartree.py`: the LiarTree online phase, extends ClusTree with noise
  handling, novelty growth, and overlap reduction.
- `src/noise_handler.py`: the LiarTree noise math (noise probability, density,
  and the Gompertz noise-to-cluster test).
- `src/overlap_handler.py`: the local overlap look-ahead and child-entry swap
  used during LiarTree descent.
- `src/offline.py`: the offline phase, re-clusters the leaf micro-clusters with
  scikit-learn k-Means or DBSCAN.
- `src/stream_runner.py`: baseframework integration (dataset loading, CMM and
  Temporal Silhouette metrics) plus the `StreamRunner` that streams a dataset
  through a tree, handles the anytime stop, and runs the offline phase.
- `run_clustree.py`: root runner for ClusTree.
- `run_liartree.py`: root runner for LiarTree.
- `evaluation/`: parameter and stream-rate sweep scripts and notebooks used for
  the experiments.
- `tests/`: unit tests for the offline phase and the overlap logic.
- `baseframework/`: provided framework code, datasets, and metrics reused by
  this implementation.

## Setup

Clone the repository with the submodule and create the local virtual
environment:

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` simply pulls in the baseframework's pinned dependencies; no
extra packages are needed.

## Usage

Run ClusTree on one of the provided baseframework datasets:

```bash
python run_clustree.py --dataset complex9
```

Run LiarTree on the same dataset:

```bash
python run_liartree.py --dataset complex9
```

Both runners stream the full dataset, print progress while running, and report
the baseframework metrics after the offline phase. Press `Ctrl+C` at any time to
stop the stream early and still get the offline result.

Examples with explicit parameters:

```bash
python run_clustree.py \
	--dataset complex9 \
	--interrupt time \
	--descent depthfirst \
	--offline kmeans --n-clusters 9

python run_liartree.py \
	--dataset complex9 \
	--noise-threshold 0.5 \
	--offline kmeans --n-clusters 9
```

Available datasets:

- `complex9`
- `stream_densired`
- `rbf3`
- `har`

### Anytime interruption

The online insertion is interruptible, as in the paper. Both runners expose the
mode:

- `--interrupt time` (default, the paper's behavior): each insertion stops when
  the next object arrives (its arrival time is used as the deadline) or after
  `--time-slice` seconds. This is what makes it usable on a live stream.
- `--interrupt budget`: stop after a fixed number of distance computations
  (`--budget`). Deterministic and machine-independent, useful for reproducible
  runs.

`--pps` feeds the stream at a real rate (points per second); in time mode it
sets each object's deadline. `--stop-after N` stops after `N` seconds and runs
the offline phase (`-1` = whole dataset).

### Common parameters (both runners)

- `--dataset`: dataset to stream (see the list above).
- `--budget`: distance computations per object for the budget interruption mode
  (default `36` for ClusTree; `72` for LiarTree, whose overlap look-ahead does
  more distance work per node).
- `--pps`: feed rate in points per second (`0` = as fast as possible).
- `--stop-after`: stop after N seconds and run the offline phase (`-1` = all).
- `--descent`: `depthfirst` (default), `breadthfirst`, `bestfirst`, or
  `iterativedf`.
- `--max-fanout`, `--leaf-max`: inner-node and leaf capacities (default `3`).
- `--lambda`: decay rate. Default `0.001`.
- `--horizon`: snapshot interval used by the discard rule. Default `1000`.
- `--max-height`: cap on tree height (`0` = unbounded); once reached, the model
  size stops growing.
- `--no-scale`: disable the baseframework's min-max scaling.

### Fast-stream aggregation (both runners)

- `--aggregate`, `--max-aggregates`: enable the fast-stream aggregation, in which
  a separate thread groups similar objects into aggregates while insertion runs,
  so each insertion can descend deeper on a very fast stream.

### LiarTree-only parameters (`run_liartree.py`)

- `--noise-threshold`: noise probability threshold (Def. 2); lower is more
  aggressive noise detection, and a value above `1` disables noise handling.
  Default `0.5`.
- `--no-overlap`: disable the local overlap reduction, falling back to
  ClusTree's greedy entry selection.

### Offline phase and metrics (both runners)

- `--offline`: `kmeans` (default) or `dbscan`.
- `--n-clusters`: `k` for k-Means.
- `--eps`, `--min-samples`: DBSCAN parameters.
- `--unweighted`: do not weight the offline clustering by micro-cluster weight.
- `--seed`: random seed for the offline clustering.
- `--no-cmm`, `--no-ts`: skip the CMM or Temporal Silhouette metric.
- `--quiet`: suppress the progress lines while streaming.

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Data

The large datasets are not part of this repository. Download them and place the files in `data/`:

- KDD Cup 1999: https://archive.ics.uci.edu/dataset/130/kdd+cup+1999+data
- Covertype: https://archive.ics.uci.edu/dataset/31/covertype
- Synthetic datasets (e.g. spiral) are generated by `evaluation/gen_data.ipynb`.
