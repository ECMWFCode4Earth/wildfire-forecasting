import glob
import xarray as xr
import gc
import pickle

files = glob.glob('../data/**/*.*', recursive=True)

data_list = {}
for file in files:
    ds = xr.open_dataset(file)
    data_list[file[8:]] = ds[list(ds.data_vars)[0]]

# data_list.keys()
# print(*list(data_list.values()), sep='\n\n######\n\n')

p_list = {x: [] for x in data_list.keys()}

interp = data_list['reanalysis/severity.nc4'][0]

# Invariant

for f in ['cover.nc4', 'fuel.nc4', 'interim/tp.nc4']:
    p_list[f] = []
    p_list[f].append(data_list[f].interp_like(interp))

# Stage

data_list['stage.nc4'] = data_list['stage.nc4'].sel(
    time=data_list['stage.nc4'].time.dt.month.isin([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

data_list['stage.nc4'] = data_list['stage.nc4'].sortby(data_list['stage.nc4'].time.dt.month)

tm = data_list['stage.nc4'].time
p_list['stage.nc4'] = []
for t in tm:
    p_list['stage.nc4'].append(data_list['stage.nc4'].sel(time=t.time).interp_like(interp))

# Interim

for f in ['interim/RH.nc4', 'interim/t2m.nc4', 'interim/ws10m.nc4']:
    data_list[f] = data_list[f].sel(time=data_list[f].time.dt.hour.isin(range(11, 13)))
    data_list[f]['time'] = data_list[f].indexes['time'].normalize()
    data_list[f] = data_list[f].sel(time=data_list[f].time.dt.month.isin([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))


gc.collect()
tm = data_list['interim/RH.nc4'].time
i = 0
for f in ['interim/RH.nc4', 'interim/t2m.nc4', 'interim/ws10m.nc4']:
    p_list[f] = []
    for t in tm:
        p_list[f].append(data_list[f].sel(time=t.time).interp_like(interp))
        i += 1
    print(i)

# Reanalysis

for f in ['reanalysis/danger.nc4', 'reanalysis/fwi.nc4', 'reanalysis/severity.nc4']:
    p_list[f] = []
    p_list[f].append(data_list[f])

# List of months

month_list = data_list['reanalysis/severity.nc4'].time.dt.month.values

# Pickling

with open('data.pkl', 'wb') as f:
    pickle.dump([p_list, month_list], f, protocol=4)
