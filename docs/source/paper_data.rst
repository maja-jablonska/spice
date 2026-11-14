Paper data
==========

The grids and model outputs behind the results of the SPICE paper
(`arXiv:2511.10998 <https://arxiv.org/abs/2511.10998>`_) are published as the
public Hugging Face dataset
`mjablonska/spice-paper-data <https://huggingface.co/datasets/mjablonska/spice-paper-data>`_.
The revision used for the paper is tagged ``v1.1``. The scripts and notebooks
that use these files are in this repository under ``tutorial/paper_results/``.

Downloading
-----------

From the root of a clone of the repository:

.. code-block:: bash

    pip install "stellar-spice[huggingface]"
    python tutorial/paper_results/fetch_paper_data.py              # everything (~250 MB)
    python tutorial/paper_results/fetch_paper_data.py --only kochukhov

The script places every file at the path the paper scripts read it from and
skips files that are already present.

Contents
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset folder
     - Contents and local destination
   * - ``kochukhov2017_fe_lte/``
     - LTE specific-intensity grid for the spotted-star test of Kochukhov (2017)
       (paper Sect. 3.1): MARCS 8000 K, log g 4.0 model, iron varied on 17 nodes
       A(Fe) = 8.0–9.6, 20 viewing angles, 5000–5020 Å. Saved to
       ``data/kochukhov2017_fe_lte.zarr`` and read with
       :class:`~spice.spectrum.lazy_zarr_interpolator.IntensityLazyZarrInterpolator`
       in ``tutorial/paper_results/spots/kochukhov2017_fe_lte.ipynb``. The
       Turbospectrum configuration that generates it is
       ``tutorial/paper_results/spots/config_kochukhov2017_fe_lte.json``.
   * - ``dcep/literature_models/``
     - The δ Cephei literature models (paper Sect. 3.2), saved to
       ``tutorial/paper_results/cepheid/dcep_out/``; read by
       ``dcep_lit_pfactor_errors.py`` (projection factor and its errors).
   * - ``dcep/pfactor_synthesis/``
     - Disc-integrated spectra for the synthetic projection factor (paper
       Table 6, Fig. 12), saved under
       ``tutorial/paper_results/cepheid/pfactor_v11/``.
   * - ``hbeta/``
     - Turbospectrum and emulator spectra over Hβ (paper Appendix F,
       Figs. F1–F2), saved to
       ``tutorial/paper_results/emulator_validation/results/interp/hbeta/``;
       read by ``make_paper_hbeta_figures.py``.

The dataset card on Hugging Face describes each file in detail.
