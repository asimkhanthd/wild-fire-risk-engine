import os
import rasterio

import numpy as np
import matplotlib.pyplot as plt

from setup import *
from pathlib import Path

ROTHERMEL_MAP = {
    1111: 4, 1112: 9, 
    1121: 4, 1211: 4, 
    1212: 9, 1221: 4, 
    1222: 10, 1301: 4,
    21: 5, 22: 4, 
    23: 4, 31: 3, 
    32: 3, 33: 3, 
    41: 3, 42: 3,
    51: 4, 52: 4, 
    53: 3, 61: 0, 
    62: 5, 7: 0
}
FINAL_MAP = {
    1: 3, 2: 1, 3: 4, 
    4: 5, 5: 3, 6: 4, 
    7: 5, 8: 2, 9: 3, 
    10: 4, 11: 4, 
    12: 4, 13: 5,
}

def fmt(archivo_lectura:str|Path,output_folder=Path('OUTPUT') ,file_name:str='FMT',export_image:bool=False):

    if isinstance(archivo_lectura,str):
        archivo_lectura=Path(archivo_lectura)

    with rasterio.open(archivo_lectura) as src:
        fmt_eu = src.read(1).astype('float32')
        meta = src.meta.copy()

    fmt_rothermel = np.zeros_like(fmt_eu, dtype='int32')
    for key, value in ROTHERMEL_MAP.items():
        fmt_rothermel[fmt_eu == key] = value

    fmt_final = np.zeros_like(fmt_rothermel, dtype='int32')
    for key, value in FINAL_MAP.items():
        fmt_final[fmt_rothermel == key] = value

    
    fig1,ax1 = default_imshow(fmt_final,'Fuel Model Type Risk Map')

    if export_image:

        meta.update(dtype='int32', nodata=-9999, count=1, driver='GTiff')
        save_file(fmt_final, file_name, output_folder, meta, extensions=['tif','png'], fig=fig1,meta_intact=True)

    return fmt_final