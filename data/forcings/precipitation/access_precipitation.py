import cdsapi

c = cdsapi.Client()

c.retrieve(
    'sis-european-energy-sector',
    {
        'variable': 'precipitation',
        'time_aggregation': '1_year_average',
        'vertical_level': '0_m',
        'bias_correction': 'bias_adjustment_based_on_gamma_distribution',
        'format': 'zip',
    },
    'download.zip')
