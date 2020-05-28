#!/usr/bin/env python3
import calendar
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
for year in range(2006, 2016):
    for month in range(1, 13):
        days = calendar.monthrange(year,month)[1]
        server.retrieve({
	    "class": "mc",
	    "dataset": "cams_gfas",
	    "date": f"{year}-{month:02}-01/to/{year}-{month:02}-{days:02}",
	    "expver": "0001",
	    "levtype": "sfc",
	    "param": "99.210",
	    "step": "0-24",
	    "stream": "gfas",
	    "time": "00:00:00",
	    "type": "ga",
	    "target": f"{year}_{month}",
	})
