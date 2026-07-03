# API Reference

Auto-generated from docstrings. For narrative explanations of *why* and *when* to use each piece, see [Concepts](concepts.md) and [Usage](usage.md) first — this page is for looking up exact signatures.

## Core

The main entry points: loading a network, computing its partition, and querying the result.

::: parx.compute_partition

::: parx.load_network

::: parx.extract_features

::: parx.Partition

::: parx.Region

## Analysis

Neuron activity, structural complexity, and geometric size statistics — see [Concepts § Outputs](concepts.md#outputs-the-partition-object) for what these numbers mean.

::: parx.analysis

## Verification

Sanity checks on a computed partition: no overlaps, full coverage, routing consistency.

::: parx.verify

## Visualization

Every plotting function takes a keyword-only `backend: Literal["plotly", "matplotlib"] = "plotly"` argument — see [Usage § Visualization](usage.md#visualization) for the tradeoffs between the two.

::: parx.viz

## I/O & Serialization

Loading training checkpoints, and saving/loading a computed `Partition` without needing Julia again.

::: parx.io.iter_state_dicts

::: parx.io_partition.save_partition

::: parx.io_partition.load_partition

## Utilities

::: parx.precompile

::: parx.list_methods

::: parx.get_method

::: parx.diagnostics.thread_info

::: parx.diagnostics.benchmark_method
