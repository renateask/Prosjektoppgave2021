import os
import shutil
import numpy as np

def tile_xy(tile_name):
    pos = tile_name.split('_')[1].split('.')[0]
    x = int(pos.split('-')[0])
    y = int(pos.split('-')[1])
    return x,y

def fetch_sorted_tiles(path_to_tiles):
    tile_names = [tile for tile in os.listdir(path_to_tiles) if tile.endswith(".tif")]
    tile_pixel_values = [*range(0,833,64)]
    pixel_index_map = {}
    for idx,pixel in enumerate(tile_pixel_values):
        pixel_index_map[pixel] = idx
    tiles = np.empty((13,14), dtype=object)
    for tile in tile_names:
        x,y = tile_xy(tile)
        x = pixel_index_map[x]
        y = pixel_index_map[y]
        tiles[x][y] = tile
    tiles = tiles.flatten()          
    return tiles

if __name__ == '__main__':
    img_dir = "images/"
    ordered_tiles_dir = "ordered_tiles/"
    image_dir = [f for f in os.listdir("images") if not f.startswith('.')]
    if not os.path.exists(ordered_tiles_dir):
        for dir in image_dir:
            root_dir = ordered_tiles_dir+dir
            tiles_dir = img_dir + dir + "/tiled_images/"
            tiles = fetch_sorted_tiles(tiles_dir)
            os.makedirs(root_dir)
            count = 1
            for tile in tiles:
                old_dir = tiles_dir+tile
                new_dir = root_dir+'/'+str(count)+'.tif'
                shutil.copy(old_dir,new_dir)
                count += 1