import geopandas as gpd
from shapely.geometry import Polygon, box
import numpy as np
from itertools import product
import argparse
import os
##############################
# script provided by ADEBOWALE DANIEL ADEBAYO, github: https://github.com/adebowaledaniel
############################

parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input", type=str, help="Input file", required=True)
parse.add_argument("-o", "--output", type=str, help="Output file", required=True)

args = parse.parse_args()

input_shapefile = args.input
output_dir = args.output

ext = gpd.read_file(input_shapefile)

crs_ = ext.crs

### This is specifically for this case
# get the height using the perimeter of a square equation.
h = (int(ext.geometry.length) / 5) / 3 # 2 and 5

print(int(ext.geometry.length))
print("h", h)


def make_grid(polygon, edge_size):
    """
    polygon : shapely.geometry
    edge_size : length of the grid cell
    source: https://stackoverflow.com/a/68778560/9948817
    """

    bounds = polygon.bounds
    x_coords = np.arange(bounds[0] + edge_size / 2, bounds[2], edge_size)
    y_coords = np.arange(bounds[1] + edge_size / 2, bounds[3], edge_size)
    combinations = np.array(list(product(x_coords, y_coords)))
    squares = gpd.points_from_xy(combinations[:, 0], combinations[:, 1]).buffer(edge_size / 2, cap_style=3)
    return gpd.GeoSeries(squares[squares.intersects(polygon)], crs=crs_)

grid = make_grid(ext.geometry[0], h)

# offfset the grid
for i in range(len(grid)):
    grid.iloc[i] = grid.iloc[i].buffer(-500)
    # print(grid.iloc[i])

# save the grid to file as shapefile
grid.to_file(os.path.join(output_dir, "grid.shp"))

# create an empty Geoseries
grid_gdf = gpd.GeoSeries(crs=crs_)

h_2 = (int(grid.geometry.length[0]) / 1) / 3 # 1 and 5

print("h_2", h_2)

for i in range(len(grid.geometry)):
    tile = make_grid(grid.geometry[i], h_2)
    grid_gdf = grid_gdf.append(tile, ignore_index=True)

# save the tiles to file as shapefile
grid_gdf.to_file(os.path.join(output_dir, "tiles.gpkg"))
