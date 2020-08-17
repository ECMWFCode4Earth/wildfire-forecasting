# Forecasting Wildfire Danger Using Deep Learning

[![Documentation Status](https://readthedocs.org/projects/wildfire-forecasting/badge/?version=latest)](https://wildfire-forecasting.readthedocs.io/en/latest/?badge=latest)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/esowc/wildfire-forecasting/master)


The current Global ECMWF Fire Forecasting (GEFF) is based on empirical models implemented in FORTRAN several decades back. To take advantage of developments in GIS & Machine Learning and increase forecasting coverage, these models need to be adopted to Deep Learning based prediction techniques. 

The project intends to reproduce the Fire Forecasting capabilities of GEFF using Deep Learning and develop further improvements in accuracy, geography and time scale through inclusion of additional variables or optimisation of model architecture & hyperparameters. Finally, a preliminary fire spread prediction tool is proposed to allow monitoring activities.

## TL; DR
This codebase (and this README) is a work-in-progress. We are constantly refactoring and introducing breaking changes. Here's a quick few steps to get going:

* Clone the repo and create a conda environment using `minimal_environment.yml` on Ubuntu 18.04 and 20.04 only.
* Check the EDA notebooks titled [`EDA_XXX_mini_sample.ipynb`](data/EDA). We recommend `jupyterlab`.
* The notebooks also include code to download mini-samples of the dataset (`~17GiB`).
* Check the Inference notebook titled [`Sample_Inference_4_10.ipynb`](examples/Sample_Inference_4_10.ipynb).

For a deeper dive, read the instructions below or head straight to [`Code_Structure_Overview.md`](Code_Structure_Overview.md) and then explore your way around [`train.py`](src/train.py), [`test.py`](src/test.py), [`dataloader/`](src/dataloader/) and [`model/`](src/model/).

The work-in-progress documentation can be viewed online on [wildfire-forecasting.readthedocs.io](https://wildfire-forecasting.readthedocs.io/en/latest/).

## Getting Started:

- **Clone this repo**:
<br> `git clone https://github.com/esowc/wildfire-forecasting.git`
<br> `cd wildfire-forecasting`

* **Using conda**: To create the environment, run
<br> `conda env create -f environment.yml`
<br> `conda activate wildfire-dl`
    >The setup is tested on Ubuntu 18.04 only and will not work on any non-Linux systems. See [this](https://github.com/conda/conda/issues/7311) issue for further details.

* **Using docker**: Docker support is experimental. To create the image and container, run
<br> `docker build -t deepfwi .`
<br> `docker docker run -it deepfwi`

## Running Inference
* **Examples**:<br>
  The [inference_2_1.ipynb](examples/inference_2_1.ipynb) and [inference_4_10.ipynb](examples/inference_4_10.ipynb) notebooks demonstrate the end-to-end procedure of loading data, creating model from saved checkpoint, and getting the predictions for 2 day input, 1 day output; and 4 day input, 10 day output experiments respectively.
* **Testing data**:<br>
  Ensure the access to fwi-forcings and fwi-reanalysis data. Limited sample data is available at `gs://deepfwi-mini-sample` (Released for educational purposes only).
* **Obtain pre-trained model**:<br>
  Place the model checkpoint file somewhere in your system and note the filepath.
  * Checkpoint file for 2 day input, 1 day FWI prediction is available [here](src/model/checkpoints/pre_trained/2_1/epoch_41_100.ckpt)
  * Checkpoint file for 4 day input, 10 day FWI prediction is available [here](src/model/checkpoints/pre_trained/4_10/epoch_99_100.ckpt)
* **Run the inference script**:<br>
  * Set `$FORCINGS_DIR` and `$REANALYSIS_DIR` or pass the directory paths through the arguments.
  `python src/test.py -in-days=2 -out-days=1 -forcings-dir=${FORCINGS_DIR} -reanalysis-dir=${REANALYSIS_DIR} -checkpoint-file='path/to/checkpoint'`

## Implementation overview
* The entry point for training is [src/train.py](src/train.py)
  * **Example Usage**: `python src/train.py [-h] [-in-days 4] [-out-days 1] [-forcings-dir ${FORCINGS_DIR}] [-reanalysis-dir ${REANALYSIS_DIR}]`

* The entry point for inference is [src/test.py](src/test.py)
  * **Example Usage**: `python src/test.py [-h] [-in-days 4] [-out-days 1] [-forcings-dir ${FORCINGS_DIR}] [-reanalysis-dir ${REANALYSIS_DIR}] [-checkpoint-file]`

* **Configuration Details**:
<br> Optional arguments (default values indicated below):

    `  -h, --help show this help message and exit`
<pre>    -init-features 16                       Architecture complexity [int]
    -in-days 4                              Number of input days [int]
    -out-days 1                             Number of output days [int]
    -epochs 100                             Number of training epochs [int]
    -learning-rate 0.001                    Maximum learning rate [float]
    -batch-size 1                           Batch size of the input [int]
    -split 0.2                              Test split fraction [float]
    -use-16bit True                         Use 16-bit precision for training (train only) [Bool]
    -gpus 1                                 Number of GPUs to use [int]
    -optim one_cycle                        Learning rate optimizer: one_cycle or cosine (train only) [str]
    -dry-run False                          Use small amount of data for sanity check [Bool]
    -case-study False                       The case-study region to use for inference: australia,california, portugal, siberia, chile, uk [Bool/str]
    -clip-output False                      Limit the inference to the output values within supplied range (e.g. 0.5,60) [Bool/list]
    -boxcox 0.1182                          Apply boxcox transformation with specified lambda while training and the inverse boxcox transformation during the inference. [Bool/float]
    -binned "0,5.2,11.2,21.3,38.0,50"       Show the extended metrics for supplied comma separated binned FWI value range [Bool/list]
    -undersample False                      Undersample the datapoints having smaller than specified FWI (e.g. -undersample=10) [Bool/float]
    -round-to-zero False                    Round off the target values below the specified threshold to zero [Bool/float]
    -date_range False                       Filter the data with specified date range. E.g. 2019-04-01,2019-05-01 [Bool/float]
    -cb_loss False                          Use Class-Balanced loss with the supplied beta parameter [Bool/float]
    -chronological_split False              Do chronological train-test split in the specified ratio [Bool/float]
    -model unet_tapered                     Model to use: unet, unet_downsampled, unet_snipped, unet_tapered, unet_interpolated [str]
    -out fwi_reanalysis                     Output data for training: fwi_forecast or fwi_reanalysis [str]
    -forecast-dir ${FORECAST_DIR}           Directory containing forecast data. Alternatively set $FORECAST_DIR [str]
    -forcings-dir ${FORCINGS_DIR}           Directory containing forcings data. Alternatively set $FORCINGS_DIR [str]
    -reanalysis-dir ${REANALYSIS_DIR}       Directory containing reanalysis data. Alternatively set $REANALYSIS_DIR [str]
    -smos-dir ${SMOS_DIR}                   Directory containing soil moisture data. Alternatively set $SMOS_DIR [str]
    -mask src/dataloader/mask.npy           File containing the mask stored as the numpy array [str]
    -benchmark False                        Benchmark the FWI-Forecast data against FWI-Reanalysis [Bool]
    -comment Comment of choice!             Used for logging [str]
    -checkpoint-file                        Path to the test model checkpoint [Bool/str]</pre>

* The [src/](src) directory contains the architecture implementation.
  * The [src/dataloader/](src/dataloader) directory contains the implementation specific to the training data.
  * The [src/model/](src/model) directory contains the model implementation.
  * The [src/model/base_model.py](src/model/base_model.py) script has the common implementation used by every model.

* The [data/EDA/](data/EDA/) directory contains the Exploratory Data Analysis and Preprocessing required for each dataset demonstrated via Jupyter Notebooks.
  * Forcings data: [data/EDA/fwi_forcings.ipynb](data/EDA/fwi_forcings.ipynb) (*Resolution: 0.07 deg x 0.07 deg, 10 days*)
  * Reanalysis data: [data/EDA/fwi_reanalysis.ipynb](data/EDA/fwi_reanalysis.ipynb) (*Resolution: 0.1 deg x 0.1 deg, 1 day*)
  * Forecast data: [data/EDA/fwi_forecast.ipynb](data/EDA/fwi_forecast.ipynb) (*Resolution: 0.1 deg x 0.1 deg, 10 days*)
  
  To-Fix - 
  * Soil moisture data: [data/EDA/soil_moisture.ipynb](data/EDA/soil_moisture.ipynb) (*Resolution: 600x1440, 3 days*)
  * FRP data: [data/EDA/frp.ipynb](data/EDA/frp.ipynb) (*Resolution: 0.1 deg x 0.1 deg, 1 day*)
  
* A walk-through of the codebase is in the [Code_Structure_Overview.md](Code_Structure_Overview.md).

## Documentation

We use Sphinx for building our docs and host them on Readthedocs. The latest build of the docs can be accessed online [here](https://wildfire-forecasting.readthedocs.io/en/latest/). In order to build the docs from source, you will need `sphinx` and `sphinx-autoapi`. Follow the instructions below:

```bash
cd docs
make html
```

Once the docs get built, you can access them inside [`docs/build/html/`](docs/build/html/index.html).

### Acknowledgements

This project tackles [Challenge #26](https://github.com/esowc/challenges_2020/issues/10) from Stream 2: Machine Learning and Artificial Intelligence, as part of the [ECMWF Summer of Weather Code 2020](https://esowc.ecmwf.int/) Program.

Team: Roshni Biswas, Anurag Saha Roy, Tejasvi S Tomar.