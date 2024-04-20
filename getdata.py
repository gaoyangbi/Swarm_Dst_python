from viresclient import SwarmRequest
from datetime import datetime as dt


# Set up connection with server
request = SwarmRequest(
url="https://vires.services/ows",
token="gGHijZm0xmMw5QWGqnPz7wwY9NQt8vod")
# Set collection to use
# - See https://viresclient.readthedocs.io/en/latest/available_parameters.html
request.set_collection("SW_OPER_MAGC_LR_1B")
# Set mix of products to fetch:
#  measurements (variables from the given collection)
#  models (magnetic model predictions at spacecraft sampling points)
#  auxiliaries (variables available with any collection)
# Optionally set a sampling rate different from the original data
request.set_products(
    measurements=[ "B_NEC"],
    models=["'CHAOS-internal' = 'CHAOS-Core' + 'CHAOS-Static'"],
    auxiliaries=["QDLat", "QDLon","Dst"],
    sampling_step="PT1S"
    )
# Fetch data from a given time interval
# - Specify times as ISO-8601 strings or Python datetime


for year in range(2014,2015):

    for month in range(1,2):
        if month == 12:
            start = str(year) + "-" + str(month) + "-01T00:00"
            end   = str(year+1) + "-01-01T00:00"
        else:
            start = str(year) + "-" + str(month).zfill(2) + "-01T00:00"
            end   = str(year) + "-" + str(month+1).zfill(2) + "-01T00:00"
        print(start)
        print(end)
        data = request.get_between(
            start_time=start,
            end_time=end
        )
        # Load the data as an xarray.Dataset
        ds = data.as_xarray()
        name = "Swarm_Bnec_" + str(year) + str(month).zfill(2) + ".cdf"
        data.to_file(name, overwrite=True)
