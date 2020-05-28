# ![Logo](docs/geff-logo.png) Global ECMWF Fire Forecasting (GEFF) model

> Maintainer: [Francesca Di Giuseppe](https://www.ecmwf.int/en/about/who-we-are/staff-profiles/francesca-di-giuseppe), please report any issues or bugs to francesca.digiuseppe@ecmwf.int.

The Global ECMWF Fire Forecasting (GEFF) model is a Fortran program to calculate fire danger indices from atmospheric inputs.
It implements the Fire Weather Index, the National Fire Danger Rating System and the McArthur ratings in one single infrastructure.
While it was principally designed for gridded data, it can operate with any kind of inputs.

### Content

- docs/ documentation
- src/ source code
- tests/ script for launching an example run
- data/ sample input data (please note, ERA-Interim and other inputs will need to be locally available)

### Meta

- This software and functions herein are part of an experimental open-source project. They are provided as is, without any guarantee.
- Please note that this project is released with a [Contributor Code of Conduct](CONDUCT.md). By participating in this project you agree to abide by its terms.
- License: [Apache License 2.0](LICENSE)
- If you use this software, please cite the following paper:<br/>
  _Di Giuseppe, F., Pappenberger, F., Wetterhall, F., Krzeminski, B., Camia, A., Libertá, G. and San Miguel, J., 2016.
  **The potential predictability of fire danger provided by numerical weather prediction**.<br/>
  Journal of Applied Meteorology and Climatology, 55(11), pp.2469-2491. https://journals.ametsoc.org/doi/abs/10.1175/JAMC-D-15-0297.1

### Release notes

* 2019/03 GEFF 3.0:

  * Aligned version with JRC settings and parameters
  * implemented 5 latitudinal belts and not 4 for "le" parameter (DMC calculation)
  * Changed day-length adjustment factor, "Lf", equatorial belt between 10/-10 instead than 15/-15 to be coherent with the "Le"
  * Removed any stopping rule associated to snow on the ground

* 2018/04 GEFF 2.2: Reanalysis release documented in Vitolo et al (Scientific Data, under review)

* 2015/11 GEFF 2.0: First full release documented in Di Giuseppe et al (JAMC 2016)

* 2018/05 Introduced an lmask check to reset *only* NFDRS calculation on points where fuel model, climatic regions and vegetation stage might not be available

* 2017/05 Changed the way the land see mask is handled to allow for the small islands to be included

* 2017/02 GEFF 1.3: Corrected bug in the keetch byram calculation

* 2014/08 GEFF 1.2: Added the FWI calculation

* 2014/06 GEFF 1.0: This is the first release of the GEFF model.

* 2014/03 Initial version

