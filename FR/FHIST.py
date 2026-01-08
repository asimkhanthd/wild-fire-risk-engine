import os
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.mask import mask

def Fhist(folder_pre, folder_post, output_fhist,export_image:bool=False)->None:


    def reproject_raster(src_path:str, dst_crs:str = "EPSG:32629")->tuple[np.ndarray, dict]:

        with rasterio.open(src_path) as src:

            transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})

            dest_array = np.empty((height, width), dtype=src.dtypes[0])

            reproject(source=rasterio.band(src, 1), destination=dest_array,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)
            
            return dest_array, kwargs

    def calcular_dnbr(b8_may, b12_may, b8_oct, b12_oct, idx):
        year = 2016 + idx
        
        # Reproyectar todo en memoria sin guardar intermedios
        nir_m, meta_m = reproject_raster(b8_may)
        swir_m, _ = reproject_raster(b12_may)
        nir_o, _ = reproject_raster(b8_oct)
        swir_o, meta_o = reproject_raster(b12_oct)

        # Calcular NBR e dNBR en memoria
        nir_m = nir_m.astype('float32')
        swir_m = swir_m.astype('float32')
        nir_o = nir_o.astype('float32')
        swir_o = swir_o.astype('float32')
        
        np.seterr(divide='ignore', invalid='ignore')
        nbr_pre = (nir_m - swir_m) / (nir_m + swir_m)
        nbr_post = (nir_o - swir_o) / (nir_o + swir_o)
        dnbr = nbr_pre - nbr_post

        # Reclasificación directa en memoria
        recl = np.where(dnbr < 0.27, 0, 1).astype('int32')
        
        # Leer geometrías para máscara
        historico = gpd.read_file(rf'C:\Users\Mateo G\Desktop\STORCITO\Fotos\HIST\Historico_incendios\hist_{year}.shp')
        buff = historico.geometry.buffer(660)
        buff_gdf = gpd.GeoDataFrame({'geometry': buff}, crs=historico.crs)
        mask_gdf = buff_gdf.dissolve()
        geometries = [g for g in mask_gdf.geometry]

        # Crear raster enmascarado en memoria usando MemoryFile
        from rasterio.io import MemoryFile
        with MemoryFile() as memfile:
            with memfile.open(driver='GTiff', height=recl.shape[0], width=recl.shape[1], count=1,
                            dtype=recl.dtype, crs=meta_o['crs'], transform=meta_o['transform']) as mem_src:
                mem_src.write(recl, 1)
            with memfile.open() as mem_src:
                out_image, out_transform = mask(mem_src, geometries, crop=True)
        
        return out_image, out_transform, meta_o, year

    pre8 = sorted([os.path.join(folder_pre, f) for f in os.listdir(folder_pre) if f.endswith('B8A_(Raw).tiff')])
    pre12 = sorted([os.path.join(folder_pre, f) for f in os.listdir(folder_pre) if f.endswith('B12_(Raw).tiff')])
    post8 = sorted([os.path.join(folder_post, f) for f in os.listdir(folder_post) if f.endswith('B8A_(Raw).tiff')])
    post12 = sorted([os.path.join(folder_post, f) for f in os.listdir(folder_post) if f.endswith('B12_(Raw).tiff')])

    # Procesar todos los años acumulando en memoria
    suma_total = None
    target_meta = None
    
    for i in range(min(len(pre8), len(pre12), len(post8), len(post12))):
        out_image, out_transform, meta, year = calcular_dnbr(pre8[i], pre12[i], post8[i], post12[i], i)
        
        if suma_total is None:
            # Primera imagen - usar como referencia
            target_shape = out_image.shape[1:]
            suma_total = np.zeros(target_shape, dtype='float32')
            target_meta = meta.copy()
            target_meta.update({'transform': out_transform})
        
        # Remuestrear si es necesario y acumular
        if out_image.shape[1:] == suma_total.shape:
            suma_total += out_image[0].astype('float32')
        else:
            dest = np.zeros(suma_total.shape, dtype='float32')
            reproject(source=out_image[0].astype('float32'), destination=dest,
                      src_transform=out_transform, src_crs=meta['crs'],
                      dst_transform=target_meta['transform'], dst_crs=target_meta['crs'],
                      resampling=Resampling.nearest)
            suma_total += dest

    if suma_total is None:
        print("No historical data found.")
        return

    # Reclasificación final
    vmax = np.max(suma_total)
    if vmax <= 0:
        print("Mapa vacío (sin incendios).")
        reclas = np.zeros_like(suma_total, dtype='int32')
    else:
        interval = float(vmax) / 5.0
        bins = [0, interval, 2*interval, 3*interval, 4*interval]
        reclas = np.digitize(suma_total, bins=bins).astype('int32')

    # Mostrar imágenes desde datos en memoria
    plt.figure()
    plt.imshow(reclas, cmap='Reds', interpolation='none')
    plt.colorbar(ticks=[1,2,3,4,5], label='Risk')
    plt.title('Historical Burned Areas Risk Map')
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(suma_total, cmap='Reds')
    plt.colorbar()
    plt.title('Suma Total (histórico)')
    plt.show()
    plt.close()

    # Guardar si el usuario lo solicita
    if export_image:

        rasters_dir = r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\re'
        png_dir = r'C:\Users\Mateo G\Desktop\STORCITO\Salida Datos\HIST'
        os.makedirs(rasters_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        base_out = os.path.splitext(os.path.basename(output_fhist))[0]
        
        # Guardar suma_total como float tif
        suma_path_tif = os.path.join(rasters_dir, 'suma_total.tif')
        tmeta = target_meta.copy()
        tmeta.update(dtype='float32', count=1)
        with rasterio.open(suma_path_tif, 'w', **tmeta) as dst:
            dst.write(suma_total, 1)
        print(f"Suma total guardada: {suma_path_tif}")

        # Guardar reclasificado como int tif
        recl_path_tif = os.path.join(rasters_dir, f'{base_out}.tif')
        recl_meta = target_meta.copy()
        recl_meta.update(dtype='int32', count=1)
        with rasterio.open(recl_path_tif, 'w', **recl_meta) as dst:
            dst.write(reclas, 1)
        print(f"Reclasificado guardado: {recl_path_tif}")

        # Guardar PNG desde datos en memoria
        png_reclas = os.path.join(png_dir, f'{base_out}.png')
        plt.figure()
        plt.imshow(reclas, cmap='Reds', interpolation='none')
        plt.colorbar(ticks=[1,2,3,4,5], label='Risk')
        plt.title('Historical Burned Areas Risk Map')
        plt.savefig(png_reclas, dpi=300, bbox_inches='tight')
        plt.close()
        
        png_sum = os.path.join(png_dir, 'suma_total.png')
        plt.figure()
        plt.imshow(suma_total, cmap='Reds')
        plt.colorbar()
        plt.title('Suma Total (histórico)')
        plt.savefig(png_sum, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PNGs guardados en: {png_dir}")

        # Guardar también en output_fhist para compatibilidad
        try:
            with rasterio.open(output_fhist, 'w', **recl_meta) as dst:
                dst.write(reclas, 1)
        except Exception:
            pass

        print('Historical Burned Areas Layer completed and saved.')
    else:
        print('Historical Burned Areas Layer completed without saving.')

    return
