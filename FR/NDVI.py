import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from setup import *
from pathlib import Path

def ndvi(b4:str|Path,b8:str|Path,output_folder:str='OUTPUT',export_image:bool=False)->tuple[np.ndarray,np.ndarray]:
    """_summary_

    Args:
        b4 (str | Path): _description_
        b8 (str | Path): _description_
        output_folder (str, optional): _description_. Defaults to 'OUTPUT'.
        export_image (bool, optional): _description_. Defaults to False.

    Returns:
        tuple[np.ndarray,np.ndarray]: _description_
    """

    b4=Path(b4)
    b8=Path(b8)

    np.seterr(divide='ignore', invalid='ignore')

    with rasterio.open(b4) as src_b3:
        band4 = src_b3.read(1).astype('float32')
        meta_ref = src_b3.meta.copy()
    with rasterio.open(b8) as src_b8:
        band8 = src_b8.read(1).astype('float32')
    
    mini_info=parse_filename(b4.name)
    name_id=mini_info.id

    ndvi = np.array( (band8 - band4) / (band8 + band4) )
    
    condiciones = [
        (ndvi <= 0.27,
        (ndvi > 0.27) & (ndvi <= 0.40),
        (ndvi > 0.40) & (ndvi <= 0.54),
        (ndvi > 0.54) & (ndvi <= 0.67),
        ndvi > 0.67) 
        ]
    
    valores = [5, 4, 3, 2, 1]

    reclasificado = np.select(condiciones, valores, default=0).astype('int32')
    
    fig1,ax1=default_imshow(ndvi,'NDVI')
    fig2,ax2=default_imshow(reclasificado,'NDVI Risk Map')
    
    if export_image:
    
        save_file(ndvi, name_id, output_folder, meta_ref, 
                  'NDVI',extensions=['tif','tiff','png'], fig=fig1)
        save_file(reclasificado, name_id, output_folder, meta_ref, 
                  'NDVI_Risk_Map',extensions=['tif','tiff','png'], fig=fig2)


    return ndvi,reclasificado

def NDVI_folder(input_folder:str='INPUT',output_folder:str='OUTPUT',export_image:bool=False)->None:
    """_summary_

    Args:
        input_folder (str, optional): _description_. Defaults to 'INPUT'.
        output_folder (str, optional): _description_. Defaults to 'OUTPUT'.
        export_image (bool, optional): _description_. Defaults to False.
    """
    bandas_requeridas=["B04","B08"]

    valids,_=check_valid_entries(bandas_requeridas,input_folder=input_folder)
  
    info=read_and_group(valids)
      
    np.seterr(divide='ignore', invalid='ignore')

    ndvi =np.array([(info['B08'][i] - info['B04'][i]) / (info['B08'][i] + info['B04'][i]) 
           for i in range(len(info['id']))])

    condiciones = [
        (ndvi <= 0.27,
        (ndvi > 0.27) & (ndvi <= 0.40),
        (ndvi > 0.40) & (ndvi <= 0.54),
        (ndvi > 0.54) & (ndvi <= 0.67),
        ndvi > 0.67) 
        ]
    
    valores = [5, 4, 3, 2, 1]

    reclasificados = np.select(condiciones, valores, default=0).astype('int32')

    if export_image:
        
        for ndvi_i,meta_ref_i,extra_info in zip(ndvi,info['meta_ref'],info['id']): 

            fig1,ax1=default_imshow(ndvi_i,'NDVI')
            save_file(ndvi_i, extra_info, output_folder, meta_ref_i, 'NDVI',extensions=['tif','tiff','png'], fig=fig1)
           
        for reclasificado_i,meta_ref_i,extra_info in zip(reclasificados,info['meta_ref'],info['id']):

            fig1,ax1=default_imshow(reclasificado_i,'NDVI Risk Map')
            save_file(reclasificado_i, extra_info, output_folder, meta_ref_i, 'NDVI_Risk_Map',extensions=['tif','tiff','png'], fig=fig1)
           
if __name__ == "__main__":
    NDVI_folder(export_image=True)