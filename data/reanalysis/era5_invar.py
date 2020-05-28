import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'high_vegetation_cover', 'leaf_area_index_high_vegetation', 'slope_of_sub_gridscale_orography',
            'type_of_high_vegetation',
        ],
        'year': '2010',
        'month': '04',
        'day': '01',
        'time': '12:00',
        'area': [
            23.75, -15, -40,
            48.75,
        ],
        'format': 'netcdf',
    },
    'era5_invar.nc')
