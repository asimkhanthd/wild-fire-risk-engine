import os
import rasterio

import numpy as np
import matplotlib.pyplot as plt

from setup import *
from osgeo import gdal
from pathlib import Path

def mdt(ruta_mdt, 
        salida_mdt, salida_slope, salida_aspect, show_plots=True):
    
    # XXX: Hay dos parametros de salida que no se usan: ruta_slope, ruta_aspect

    output_folder=Path('OUTPUT')
    export_image=False
    print('MDT, SLOPE and ASPECT Layers processing...')


    rasters_dir = output_folder/'TIFFs'/'MDT'
    png_dir = output_folder/'PNGs'/'MDT'
    
    if export_image:

        rasters_dir.mkdir(exist_ok=True,parents=True)
        png_dir.mkdir(exist_ok=True,parents=True)

    # leer MDT completo (masked to avoid extra nan passes)
    with rasterio.open(ruta_mdt) as src:
        mdt = src.read(1, masked=True).filled(0).astype('float32')
        meta = src.meta.copy()
    print("MDT original cargado.")

    # slope/aspect via GDAL (faster than numpy gradients)
    ds = gdal.Open(ruta_mdt)
    slope_ds = gdal.DEMProcessing('/vsimem/slope_tmp.tif', ds, 'slope', format='MEM')
    aspect_ds = gdal.DEMProcessing('/vsimem/aspect_tmp.tif', ds, 'aspect', format='MEM')
    slope = slope_ds.ReadAsArray().astype('float32')
    aspect = aspect_ds.ReadAsArray().astype('float32')

    slope_ds = aspect_ds = ds = None  # close datasets
    aspect = np.where(aspect < 0, 360 + aspect, aspect)
    print("Slope y Aspect calculados.")

    def save_and_plot(array, base_path, title):
        base = os.path.splitext(os.path.basename(base_path))[0]
        meta_out = meta.copy()
        meta_out.update(dtype='int32', count=1, nodata=-9999, driver='GTiff')
        
        if export_image:

            tif_path = rasters_dir/f'{base}.tif'

            with rasterio.open(tif_path, 'w', **meta_out) as dst:
                dst.write(array.astype('int32'), 1)

            png_path = png_dir/f'{base}.png'
            
            plt.figure(figsize=(8, 6))
            plt.imshow(array, cmap='Reds')
            plt.colorbar()
            plt.title(title)
            plt.tight_layout()

       
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
    
    # reclasificaciones
    print("Reclasificando MDT...")
    mdt_bins = [0, 200, 400, 600, 800]
    mdt_classes = np.array([0, 5, 4, 3, 2, 1], dtype='int32')
    mdt_re = mdt_classes[np.digitize(mdt, mdt_bins, right=True)]

    fig_mdt, ax_mdt = default_imshow(mdt_re, 'MDT Risk Map', {'label':'Risk'})

    print("MDT reclasificado completado.")

    print("Reclasificando Slope...")
    slope_bins = [5, 15, 25, 35]
    slope_classes = np.array([1, 2, 3, 4, 5], dtype='int32')
    slope_re = slope_classes[np.digitize(slope, slope_bins, right=True)]
    fig_slpe, ax_slope = default_imshow(slope_re, 'Slope Risk Map', {'label':'Risk'})
    print("Slope reclasificado completado.")

    print("Reclasificando Aspect...")

    conditions= [(aspect >= 0) & (aspect < 45) | (aspect == 360),
                 (aspect >= 45) & (aspect < 90),
                 (aspect >= 90) & (aspect < 135),
                 (aspect >= 135) & (aspect < 180),
                 (aspect >= 180) & (aspect < 225),
                 (aspect >= 225) & (aspect < 270),
                 (aspect >= 270) & (aspect < 315),
                 (aspect >= 315) & (aspect < 360),
    ]
    #XXX: La secuencia de choices es correcta?
    choices= [1, 2, 3, 4, 5, 5, 3, 2]
    aspect_re = np.select(conditions,choices,default=0,).astype('int32')
    fig_aspect, ax_aspect = default_imshow(aspect_re, 'Aspect Risk Map', {'label':'Risk'})
    print("Aspect reclasificado completado.")

    if export_image:
        # save_file(fig_mdt,)
        pass



    save_and_plot(mdt_re, salida_mdt, 'MDT Risk Map')
    save_and_plot(slope_re, salida_slope, 'Slope Risk Map')
    save_and_plot(aspect_re, salida_aspect, 'Aspect Risk Map')


    print("MDT, SLOPE and ASPECT Layers completed.")
    return
