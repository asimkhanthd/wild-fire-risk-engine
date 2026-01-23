import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from setup import *
from pathlib import Path

def gci(b3:str|Path,b8:str|Path,output_folder:str='OUTPUT',
        export_image:bool=False,show_plots:bool=False)->None:
    """_summary_

    Args:
        b3 (str | Path): _description_
        b8 (str | Path): _description_
        output_folder (str, optional): _description_. Defaults to 'OUTPUT'.
        show_plots (bool, optional): _description_. Defaults to False.
        export_image (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    b3=Path(b3)
    b8=Path(b8)

    np.seterr(divide='ignore', invalid='ignore')

    with rasterio.open(b3) as src_b3:
        band3 = src_b3.read(1).astype('float32')
        meta_ref = src_b3.meta.copy()
    with rasterio.open(b8) as src_b8:
        band8 = src_b8.read(1).astype('float32')
    
    mini_info=parse_filename(b3.name)
    name_id=mini_info.id

    gci = (band8 / band3) - 1
    
    fig1,ax1=default_imshow(gci,'GCI')
    
    if show_plots:
        plt.show()

    if export_image:
        save_file(gci, name_id, output_folder, meta_ref, 'GCI',extensions=['tif','tiff','png'], fig=fig1)

    return gci

def GCI_folder(input_folder:str='INPUT',output_folder:str='OUTPUT',export_image:bool=False)->None:
    """_summary_

    Args:
        input_folder (str, optional): _description_. Defaults to 'INPUT'.
        output_folder (str, optional): _description_. Defaults to 'OUTPUT'.
        export_image (bool, optional): _description_. Defaults to False.
    """

    valids,_=check_valid_entries(["B03","B08"],input_folder=input_folder)
    
    info=read_and_group(valids)
   
    np.seterr(divide='ignore', invalid='ignore')

    gci =[ (info['B08'][i] / info['B03'][i]) - 1
          for i in range(len(info['id'])) ]
    
    if export_image:

        for gci_i,meta_ref_i,extra_info in zip(gci,info['meta_ref'],info['id']):
    
            fig1,ax1=default_imshow(gci_i,'GCI')
            save_file(gci_i, extra_info, output_folder, meta_ref_i, 'TWI',extensions=['tif','tiff','png'], fig=fig1)


if __name__ == "__main__":
    GCI_folder(export_image=True)