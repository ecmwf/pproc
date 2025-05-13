# PProc

[![ESEE Production Chain](https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/production_chain_badge.svg)](https://github.com/ecmwf/codex/blob/main/ESEE#production-chain)
[![Maturity: Incubating](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/incubating_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#incubating)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/apache-2-0)
[![Latest Release](https://img.shields.io/github/v/tag/ecmwf/pproc?color=blue&filter=*.*.*&label=Release)](https://github.com/ecmwf/pproc/tags)

> \[!IMPORTANT\]
> This software is **Incubating** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

PProc is a framework to run meteorological data processing workflows operaionally at ECMWF.

## Stand-alone tools

These tools are designed to be easily used on their own.

  * pproc-interpol: Perform interpolation and regridding

## Tools intended for operational suites

These tools are designed to be integrated in operational workflows

  * pproc-accumulate: Perform generic accumulations
  * pproc-clustereps: Perform ensemble clustering (using K-Means). Wrapper around the following:
    * pproc-clustereps-pca: Principal component analysis for dimensionality reduction
    * pproc-clustereps-cluster: Perform K-Means clustering in PC space
    * pproc-clustereps-attr: Attribute clustering results to climatological regimes
  * pproc-ensms: Compute ensemble means and standard deviations
    * pproc-wind: Compute ensemble means and standard deviations on wind fields (u/v, wind speed, etc.)
  * pproc-extreme: Compute EFI and Shift of Tails
  * pproc-histogram: Compute histograms
  * pproc-probabilities: Compute probabilities
    * pproc-anomaly-probabilities: Compute probabilities on anomalies
  * pproc-pts: Compute probabilities of tropical storm based on tracks
  * pproc-quantiles: Compute quantiles from an ensemble
  * pproc-significance: Compute significance based on a Wilcoxon-Mann-Whitney test
  * pproc-thermal-indices: Compute thermal indices (see [thermofeel](https://github.com/ecmwf/thermofeel))

## License

See [LICENSE](LICENSE)

## Copyright

© 2021- ECMWF. All rights reserved.
