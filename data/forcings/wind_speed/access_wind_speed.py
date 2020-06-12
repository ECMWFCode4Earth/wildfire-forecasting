import cdsapi

c = cdsapi.Client()

c.retrieve(
    'sis-european-energy-sector',
    {
        'variable': 'wind_speed',
        'time_aggregation': '6_hour_average',
        'vertical_level': '10_m',
        'bias_correction': 'bias_adjustment_based_on_weibull_distribution',
        'format': 'zip',
    },
    'download.zip')
