import cdsapi

c = cdsapi.Client()

c.retrieve(
    'sis-european-energy-sector',
    {
        'variable': 'air_temperature',
        'time_aggregation': '6_hour_average',
        'vertical_level': '2_m',
        'bias_correction': 'normal_distribution_adjustment',
        'format': 'zip',
    },
    'download.zip')
