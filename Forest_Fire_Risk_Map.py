import sys
# Ensure the package folder containing 'rutinas' is on sys.path so imports like

sys.path.append(r'C:\Users\Mateo G\Desktop\STORCITO\Codigos\FR_Gal\FR')

import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import os
from osgeo import gdal
import FR.FMT_eu as Fmt
import FR.MDT as Mdt
import FR.IUF as Wui
import FR.infra as Infra
import FR.NDVI as Ndvi
import FR.FHIST as Fhist
import FR.FWI as Fwi
import FR.cropped as Cropped
from FR.ahp import normalize_matrix, calculate_weights, consistency_ratio


#viva la vida

# Rutas de entrada (comentadas: descomentar para ejecutar cada función individualmente)
#Mdt.mdt(r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\DTM\DTM.tif',
#        r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\MDT\MDT.tif',
#        r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\MDT\SLOPE.tif',
#        r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\MDT\ASPECT.tif')
#Ndvi.Ndvi(r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\NDVI\B4.tiff', r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\NDVI\B8.tiff')
#Fhist.Fhist(r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\HIST\Bandas_pre', r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\HIST\Bandas_post', r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\HIST\HIST.tif')
#Fmt.fmt(r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\FUELS\FMT_NationalScenario_2019.tif', r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\FMT\FMT.tif')
#Infra.infrastructure(r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\INFRA\infraestructuras_gal.shp', r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\INFRA\infra_layer.tif')
#Wui.wui(r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\INFRA\infraestructuras_gal.shp', r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\IUF\CLC_galicia.shp', r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\IUF\WUI.tif')
Fwi.f_w_index(r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\FWI', r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\FWI\FWI.tif')

# Reproyección y redimensionado de capas
print("Reproyecting and resizing layers...")
input_folder = r'C:\Users\Mateo G\Desktop\STORCITO\Fotos\Forest Fire Risk Map'
output_folder = r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\re'
os.makedirs(output_folder, exist_ok=True)

target_epsg, target_pixel_size = 32629, 20
for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        input_path = os.path.join(input_folder, filename)
        gdal.Warp(os.path.join(output_folder, filename), input_path,
                  dstSRS=f'EPSG:{target_epsg}', xRes=target_pixel_size, yRes=target_pixel_size, resampleAlg='cubic')
        print(f"  ✓ {filename}")

print("Reprojection completed.")

