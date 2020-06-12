# Forecasting Wildfire Danger Using Deep Learning

The current Global ECMWF Fire Forecasting (GEFF) is based on empirical models implemented in FORTRAN several decades back. To take advantage of developments in GIS & Machine Learning and increase forecasting coverage, these models need to be adopted to Deep Learning based prediction techniques. 

The project intends to reproduce the Fire Forecasting capabilities of GEFF using Deep Learning and develop further improvements in accuracy, geography and time scale through inclusion of additional variables or optimisation of model architecture & hyperparameters. Finally, a preliminary fire spread prediction tool is proposed to allow monitoring activities.


## Development

### Project Structure

- **`src/`**
  - **`models/`** - deep learning models in development
- **`data/`** - data access scripts and exploratory data analysis.
    - `forcings/` - sample meteorological forcings data (for eg. wind speed, temperature, precipitation, relative humidity, etc).
    - `forecast/` - sample data produced by forecast models (for eg. fwi-forecast produced by EFFIS).
    - `reanalysis/` - sample assimilated historical observation data (for eg fwi-reanalysis produced by ERA-5).
    - `GEFFv3.0` - sample input data, cloned from the Global ECMWF Fire Forecasting (GEFF) [repository](https://git.ecmwf.int/projects/CEMSF/repos/geff/browse/data).
- **`docs/`** - the most recent build of the source code documentation.

### Developer Setup

To install necessary dependencies, run:
```bash
pip install -r requirements.txt
```


### Acknowledgements
This project tackles [Challenge #26](https://github.com/esowc/challenges_2020/issues/10) from Stream 2: Machine Learning and Artificial Intelligence, @[ECMWF Summer of Weather Code 2020](https://esowc.ecmwf.int/).