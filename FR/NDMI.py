import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from FR.rutinas.setup import *
from pathlib import Path

def ndmi(b8:str|Path,b11:str|Path,output_folder:str='OUTPUT',export_image:bool=False)->None:
    """_summary_

    Args:
        b11 (str | Path): _description_
        b8 (str | Path): _description_
        output_folder (str, optional): _description_. Defaults to 'OUTPUT'.
        export_image (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    b8=Path(b8)
    b11=Path(b11)

    np.seterr(divide='ignore', invalid='ignore')

    with rasterio.open(b11) as src_b11:
        band11 = src_b11.read(1).astype('float32')
        meta_ref = src_b11.meta.copy()
    with rasterio.open(b8) as src_b8:
        band8 = src_b8.read(1).astype('float32')
    
    mini_info=parse_filename(b11.name)
    name_id=mini_info.id

    ndmi = (band8 - band11) / ( band8 + band11 )
    
    if export_image:
        fig1,ax1=default_imshow(ndmi,'ndmi')
        save_file(ndmi, name_id, output_folder, meta_ref, 'NDMI',extensions=['tif','tiff','png'], fig=fig1)

    return ndmi

def NDMI_folder(input_folder:str='INPUT',output_folder:str="OUTPUT",export_image:bool=False)->np.ndarray:
    """_summary_

    Args:
        input_folder (str, optional): _description_. Defaults to 'INPUT'.
        output_folder (str, optional): _description_. Defaults to "OUTPUT".
        export_image (bool, optional): _description_. Defaults to False.
    """

    valids,_=check_valid_entries(["B08","B11"],input_folder=input_folder)

    info=read_and_group(valids)

    np.seterr(divide='ignore', invalid='ignore')

    ndmi = np.array([(info['B08'][i] - info['B11'][i]) / (info['B08'][i] + info['B11'][i]) 
            for i in range(len(info['id'])) ])

    if export_image:

        for ndm_i,meta_ref_i,extra_info in zip(ndmi,info['meta_ref'],info['id']):
            fig1,ax1=default_imshow(ndm_i,'NDMI')
            save_file(ndm_i, extra_info, output_folder, meta_ref_i, 'NDMI',extensions=['tif','tiff','png'], fig=fig1)
    
    return ndmi

if __name__ == "__main__":

    import cProfile
    import pstats

    with cProfile.Profile() as profile:
        NDMI_folder()

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats(20)