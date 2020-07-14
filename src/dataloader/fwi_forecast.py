"""
The dataset class to be used with fwi-forcings and fwi-forecast data.
"""
from glob import glob

import xarray as xr

import torch
import torchvision.transforms as transforms

from dataloader.base_loader import ModelDataset as BaseDataset


class ModelDataset(BaseDataset):
    """
    The dataset class responsible for loading the data and providing the samples for
    training.

    :param BaseDataset: Base Dataset class to inherit from
    :type BaseDataset: base_loader.BaseDataset
    """

    def __init__(
        self,
        out_var=None,
        out_mean=None,
        forecast_dir=None,
        forcings_dir=None,
        reanalysis_dir=None,
        transform=None,
        hparams=None,
        **kwargs,
    ):
        """
        Constructor for the ModelDataset class

        :param out_var: Variance of the output variable, defaults to None
        :type out_var: float, optional
        :param out_mean: Mean of the output variable, defaults to None
        :type out_mean: float, optional
        :param forecast_dir: The directory containing the FWI-Forecast data, defaults to
            None
        :type forecast_dir: str, optional
        :param forcings_dir: The directory containing the FWI-Forcings data, defaults to
            None
        :type forcings_dir: str, optional
        :param reanalysis_dir: The directory containing the FWI-Reanalysis data,
            defaults to None
        :type reanalysis_dir: str, optional
        :param transform: Custom transform for the input variable, defaults to None
        :type transform: torch.transforms, optional
        :param hparams: Holds configuration values, defaults to None
        :type hparams: Namespace, optional
        """

        super().__init__(
            out_var=out_var,
            out_mean=out_mean,
            forecast_dir=forecast_dir,
            forcings_dir=forcings_dir,
            reanalysis_dir=reanalysis_dir,
            transform=transform,
            hparams=hparams,
            **kwargs,
        )

        # Consider only ground truth and discard forecast values
        preprocess = lambda x: x.isel(time=slice(0, 1))

        inp_files = sorted(
            sorted(glob(f"{forcings_dir}/ECMWF_FO_2019*.nc")),
            # Extracting the month and date from filenames to sort by time.
            key=lambda x: int(x.split("2019")[1].split("_1200_hr_")[0][:2]) * 100
            + int(x.split("2019")[1].split("_1200_hr_")[0][2:]),
        )[:736]
        inp_invalid = lambda x: not (
            1 <= int(x.split("2019")[1].split("_1200_hr_")[0][:2]) <= 12
            and 1 <= int(x.split("2019")[1].split("_1200_hr_")[0][2:]) <= 31
        )
        # Checking for valid date format
        assert not (sum([inp_invalid(x) for x in inp_files])), (
            "Invalid date format for input file(s)."
            "The dates should be formatted as YYMMDD."
        )
        with xr.open_mfdataset(
            inp_files, preprocess=preprocess, engine="h5netcdf"
        ) as ds:
            self.input = ds.load()

        out_files = sorted(
            glob(f"{forecast_dir}/ECMWF_FWI_2019*_1200_hr_fwi.nc"),
            # Extracting the month and date from filenames to sort by time.
            key=lambda x: int(x[-19:-17]) * 100 + int(x[-17:-15]),
        )[:184]
        out_invalid = lambda x: not (
            1 <= int(x[-19:-17]) <= 12 and 1 <= int(x[-17:-15]) <= 31
        )
        # Checking for valid date format
        assert not (sum([out_invalid(x) for x in out_files])), (
            "Invalid date format for output file(s)."
            "The dates should be formatted as YYMMDD."
        )
        with xr.open_mfdataset(
            out_files, preprocess=preprocess, engine="h5netcdf"
        ) as ds:
            self.output = ds.load()

        # Ensure timestamp matches for both the input and output
        assert len(self.input.time) == len(self.output.time)

        # Loading the mask for output variable if provided as generating from NaN mask
        self.mask = ~torch.isnan(torch.from_numpy(self.output["fwi"][0].values))

        # Mean of output variable used for bias-initialization.
        self.out_mean = out_mean if out_mean else 18.389227

        # Variance of output variable used to scale the training loss.
        self.out_var = (
            out_var if out_var else 20.80943 if self.hparams.loss == "mae" else 716.1736
        )

        # Input transforms including mean and std normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Mean and standard deviation stats used to normalize the input data to
                # the mean of zero and standard deviation of one.
                transforms.Normalize(
                    (
                        72.03445,
                        281.2624,
                        2.4925985,
                        6.5504117,
                        72.03445,
                        281.2624,
                        2.4925985,
                        6.5504117,
                    ),
                    (
                        18.8233801,
                        21.9253515,
                        6.37190019,
                        3.73465273,
                        18.8233801,
                        21.9253515,
                        6.37190019,
                        3.73465273,
                    ),
                ),
            ]
        )
