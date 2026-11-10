README G_RVS PASSBAND FOR GAIA DR3

The GRVS passband was produced by Coordination Unit 6 of the Gaia Data Processing and Analysis Consortium. It was derived by comparing the spectra of 108 stars having both high-quality, cleaned and calibrated RVS spectra and reference spectra from the CALSPEC and NGSL spectrophotometric libraries. The zero-point of this GRVS filter is 21.317+/-0.002 mag.

The passbands are found in the file grvsfilter.csv. The first column of the file contains the wavelength expressed in nm; the second column contains the GRVS transmissivity curve at the corresponding wavelength and the third column the uncertainty on the transmissivity.

Publications using GRVS Photometry should refer to the paper by Sartoretti et al. (2022), where a full description of this quantity can be found. A link to the paper by Sartoretti et al. (2022) will be provided from the page: www.cosmos.esa.int/web/gaia/dr3-papers.

Credit: ESA/Gaia/DPAC/CU6, C. Babusiaux, P. Sartoretti, O. Marchal, C. Jordi.

SPICE provenance
Retrieved 2026-09-19 from ESA's Gaia DR3 passband distribution:
https://www.cosmos.esa.int/web/gaia/dr3-passbands
https://www.cosmos.esa.int/documents/29201/8554615/GAIADR3_GRVSFILTER.zip/92e95589-a5c0-8f84-997d-7e913522e4c8

The CSV is distributed unchanged. SPICE converts column 1 from nm to
Angstroms and reads column 2 as transmission. The published 21.317 mag
zero point is used for Vega-system count-rate photometry (Sartoretti et al.,
A&A 674, A6, https://doi.org/10.1051/0004-6361/202243615, section 7).
The AB count-rate zero point is derived separately by integrating the
3631 Jy reference spectrum through this response.
