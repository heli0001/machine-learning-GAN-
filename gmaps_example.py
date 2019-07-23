import gmaps
import gmaps.datasets
from ipywidgets.embed import embed_minimal_html
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as clrs
import numpy as np
gmaps.configure(api_key="")


# This script reads in a file of lat long pulses and plots them in an html navigatable google map along with road information.

path_to_data = 'C:\\Users\\ScotSh03\\Documents\\gan\\data\\'
ds = pd.read_csv(path_to_data + 'mdexdrivin.csv')

print(ds.size)

# Look at a particular journey
da = ds[ds.journeyid == 116730701].reset_index(drop=True)

# ds = pd.read_csv(fun_path + 'dlylatlongs.csv')

fig = gmaps.figure()

db = ds[ds.journeyid == 105810081].reset_index(drop=True)

heatmap_layer = gmaps.heatmap_layer(db[['latitude','longitude']], opacity=.9, max_intensity = .0003)
fig.add_layer(heatmap_layer)

sym_layer = gmaps.symbol_layer(da[['latitude','longitude']])
fig.add_layer(sym_layer)

output_path = 'C:\\Users\\ScotSh03\\Documents\\gan\\'
embed_minimal_html(output_path + 'dex_driving.html', views=[fig])
