# Forecasting Wildfire Danger Using Deep Learning

The current Global ECMWF Fire Forecasting (GEFF) is based on empirical models implemented in FORTRAN several decades back. To take advantage of developments in GIS & Machine Learning and increase forecasting coverage, these models need to be adopted to Deep Learning based prediction techniques. 

The project intends to reproduce the Fire Forecasting capabilities of GEFF using Deep Learning and develop further improvements in accuracy, geography and time scale through inclusion of additional variables or optimisation of model architecture & hyperparameters. Finally, a preliminary fire spread prediction tool is proposed to allow monitoring activities.

## Getting Started:

* **Clone this repo**:
<br> `https://github.com/esowc/wildfire-forecasting`
<br> `cd wildfire-forecasting`

* **Install dependencies**: To create the environment, run
<br> `conda env create -f environment.yml`
<br> `conda activate wildfire-dl`

    >The setup is tested on Ubuntu 18.04 only and might experience issues on any non-Linux systems. See [this](https://github.com/conda/conda/issues/7311) issue for further details.

The above `conda recipe` does not install [`apex`](https://github.com/NVIDIA/apex). Please follow the instructions [here](https://github.com/NVIDIA/apex#quick-start) to install NVIDIA Apex which is used for 16-bit precision training.

## Running Inference

* **Examples**:<br>
  The [inference_2_1.ipynb](examples/inference_2_1.ipynb) and [inference_4_10.ipynb](examples/inference_4_10.ipynb) notebooks demonstrate the end-to-end procedure of loading data, creating model from saved checkpoint, and getting the predictions for 2 day input, 1 day forecast; and 4 day input, 10 day forecast experiments respectively.
* **Testing data**:<br>
  Ensure the access to fwi-forcings and fwi-reanalysis data.
* **Obtain pre-trained model**:<br>
  Place the model checkpoint file somewhere in your system and note the filepath.
* **Run the inference script**:<br>
  * Optionally set `$FORCINGS_DIR` and `$REANALYSIS_DIR` to override `$PWD` as the default location of data.
  `python src/test.py -in-days=2 -out-days=1 -forcings-dir=${FORCINGS_DIR:-$PWD} -reanalysis-dir=${REANALYSIS_DIR:-$PWD} -checkpoint-file='path/to/checkpoint'`

## Implementation overview

* The entry point for training is [src/train.py](src/train.py)
  * **Example Usage**: `python src/train.py [-h]
               [-init-features 16] [-in-days 4] [-out-days 1]
               [-epochs 100] [-learning-rate 0.001] [-loss mse]
               [-batch-size 1] [-split 0.2] [-use-16bit True] [-gpus 1]
               [-optim one_cycle] [-dry-run False]
               [-clip-fwi False] [-model unet_tapered] [-out fwi_reanalysis]
               [-forcings-dir ${FORCINGS_DIR:-$PWD}]
               [-reanalysis-dir ${REANALYSIS_DIR:-$PWD}]
               [-mask dataloader/mask.npy] [-thresh 9.4]
               [-comment None]`
               
* The entry point for inference is [src/test.py](src/test.py)
  * **Example Usage**: `python src/test.py [-h]
               [-init-features 16] [-in-days 4] [-out-days 1]
               [-learning-rate 0.001] [-loss mse]
               [-batch-size 1] [-split 0.2] [-use-16bit True] [-gpus 1]
               [-dry-run False] [-case-study False]
               [-clip-fwi False] [-model unet_tapered] [-out fwi_reanalysis]
               [-forcings-dir ${FORCINGS_DIR:-$PWD}]
               [-reanalysis-dir ${REANALYSIS_DIR:-$PWD}]
               [-mask dataloader/mask.npy] [-thresh 9.4]
               [-comment None] [-checkpoint-file]`

* **Configuration Details**:
<br> Optional arguments (default values indicated below):

    `  -h, --help show this help message and exit`
<pre>    -init-features 16                       Architecture complexity
    -in-days 4                              Number of input days
    -out-days 1                             Number of output channels
    -epochs 100                             Number of training epochs
    -learning-rate 0.001                    Maximum learning rate
    -loss mse                               Loss function: mae, mse
    -batch-size 1                           Batch size of the input
    -split 0.2                              Test split fraction
    -use-16bit True                         Use 16-bit precision for training (train only)
    -gpus 1                                 Number of GPUs to use
    -optim one_cycle                        Learning rate optimizer: one_cycle or cosine (train only)
    -dry-run False                          Use small amount of data for sanity check
    -case-study False                       Limit the analysis to Australian region (inference only)
    -clip-fwi False                         Limit the analysis to the data points with 0.5 < fwi < 60 (inference only)
    -test-set /path/to/pickled/list         Load test-set filenames from specified file instead of random split
    -model unet_tapered                     Model to use: unet, unet_downsampled, unet_snipped, unet_tapered
    -out fwi_reanalysis                     Output data for training: fwi_forecast or fwi_reanalysis
    -forecast-dir ${FORECAST_DIR:-$PWD}     Directory containing forecast data. Alternatively set $FORECAST_DIR
    -forcings-dir ${FORCINGS_DIR:-$PWD}     Directory containing forcings data. Alternatively set $FORCINGS_DIR
    -reanalysis-dir ${REANALYSIS_DIR:-$PWD} Directory containing reanalysis data. Alternatively set $REANALYSIS_DIR
    -mask dataloader/mask.npy               File containing the mask stored as the numpy array
    -thresh 9.4                             Threshold for accuracy: Half of output MAD
    -comment Comment of choice!             Used for logging
    -save-test-set False                    Save the test-set file names to the specified filepath 
    -checkpoint-file                        Path to the test model checkpoint</pre>
    
* The [src/](src) directory contains the architecture implementation.
  * The [src/dataloader/](src/dataloader) directory contains the implementation specific to the training data.
  * The [src/model/](src/model) directory contains the model implementation.
  * The [src/base.py](src/base.py) directory has the common implementation used by every model.

* The [data/](data) directory contains the Exploratory Data Analysis and Preprocessing required for each dataset demonstrated via Jupyter Notebooks.
  * Forcings data: [data/fwi_global/fwi_forcings.ipynb](data/fwi_global/fwi_forcings.ipynb)
  * Reanalysis data: [data/fwi_global/fwi_reanalysis.ipynb](data/fwi_global/fwi_reanalysis.ipynb)
  * Forecast data: [data/fwi_global/fwi_forecast.ipynb](data/fwi_global/fwi_forecast.ipynb)

### Acknowledgements

This project tackles [Challenge #26](https://github.com/esowc/challenges_2020/issues/10) from Stream 2: Machine Learning and Artificial Intelligence, as part of the [ECMWF Summer of Weather Code 2020](https://esowc.ecmwf.int/) Program.

Team: Roshni Biswas, Anurag Saha Roy, Tejasvi S Tomar.