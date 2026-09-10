import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

import cv2  
import numpy as np  
from tqdm import tqdm  
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure  

#tecnique = 'AINet-main'  
#method = 'AINet_CottonyBlightV2'

#tecnique = 'DRRNet-main'  
#method = 'DRRNet_CottonyBlightV2'  

#tecnique = 'YOLOv11-main'  
#method = 'YOLOv11Net'


tecnique = 'BASNet-master'    
method = 'BASNet_IguanaDatasetV2'

tecnique = 'DGNet-main'  
method = 'DGNet_IguanaDatasetV2'  

tecnique = 'BGNet-master'  
method = 'BGNet_IguanaDatasetV2'  

tecnique = 'HitNet-main'  
method = 'Hitnet_IguanaDatasetV2'  

tecnique = 'SINet-V2-main'  
method = 'SINet-V2_IguanaDatasetV2'  

tecnique = 'PlantCamo-main'  
method = 'PCNet_IguanaDatasetV2'  

tecnique = 'C2FNet-master'  
method = 'C2FNet_IguanaDatasetV2'

#tecnique = 'OCENet-main'  
#method = 'OCENet_IguanaDatasetV2'

#tecnique = 'EAMNet-main'  
#method = 'EAMNet_IguanaDatasetV2'

#tecnique = 'CTF-Net-main'  
#method = 'CTF-Net_IguanaDatasetV2'  

#tecnique = 'CHNet-main'  
#method = 'CHNet_IguanaDatasetV2'  

#tecnique = 'ARNet-main'  
#method = 'ARNet_IguanaDatasetV2'  

#tecnique = 'AGNet-main'  
#method = 'AGNet_IguanaDatasetV4'  

def get_file_with_extension(directory, filename_without_ext):  
    """Busca un archivo con cualquier extensión en el directorio dado"""  
    for ext in ['.png', '.jpg', '.jpeg']:  # Añade más extensiones si es necesario  
        potential_file = filename_without_ext + ext  
        if os.path.exists(os.path.join(directory, potential_file)):  
            return potential_file  
    return None  


def resize_to_match(img1, img2):  
    """Redimensiona img2 para que coincida con las dimensiones de img1"""  
    if img1.shape != img2.shape:  
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)  
    return img2  


for _data_name in ['IguanaDatasetV2']:  
    mask_root = f'C:/Respaldo/Henry/Proyecto Camuflaje/Datasets/{_data_name}/test/GT'  
    #mask_root = f'D:/Datasets/COD/TestDataset/{_data_name}/test/GT'  

    if tecnique == 'DGNet-main':  
        pred_root = f'./{tecnique}/lib_pytorch/results/{method}/{_data_name}/'  
    else:  
        pred_root = f'./{tecnique}/results/{method}/{_data_name}/'  

    # Obtener lista de archivos sin extensión  
    mask_names = [os.path.splitext(f)[0] for f in sorted(os.listdir(mask_root))]  

    FM = Fmeasure()  
    WFM = WeightedFmeasure()  
    SM = Smeasure()  
    EM = Emeasure()  
    M = MAE()  

    # Crear archivo para guardar métricas por imagen  
    with open(f"{method}_{_data_name}_metrics_per_image.txt", "w") as per_image_file:  
        per_image_file.write("ImageName,Smeasure,wFmeasure,MAE,adpEm,meanEm,maxEm,adpFm,meanFm,maxFm\n")  

        for base_name in tqdm(mask_names, total=len(mask_names)):  
            try:  
                # Encontrar los archivos con sus respectivas extensiones  
                mask_file = get_file_with_extension(mask_root, base_name)  
                pred_file = get_file_with_extension(pred_root, base_name)  

                if mask_file and pred_file:  
                    mask_path = os.path.join(mask_root, mask_file)  
                    pred_path = os.path.join(pred_root, pred_file)  

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
                    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  

                    if mask is None or pred is None:  
                        print(f"Error: No se pudo leer la imagen {base_name}")  
                        continue  

                    # Redimensionar pred para que coincida con mask  
                    #pred = resize_to_match(mask, pred)  
                    mask = resize_to_match(pred, mask)  

                    # Verificar que ambas imágenes tengan el mismo tamaño  
                    if mask.shape != pred.shape:  
                        print(f"Error: Las dimensiones no coinciden después del resize para {base_name}")  
                        continue  

                    # Normalizar las imágenes si es necesario (0-255 a 0-1)  
                    #mask = mask.astype(np.float32) / 255.0  
                    #pred = pred.astype(np.float32) / 255.0  

                    # Calcular métricas para la imagen actual  
                    FM.step(pred=pred, gt=mask)  
                    WFM.step(pred=pred, gt=mask)  
                    SM.step(pred=pred, gt=mask)  
                    EM.step(pred=pred, gt=mask)  
                    M.step(pred=pred, gt=mask)  

                    # Obtener resultados para la imagen actual  
                    #fm = FM.get_results()["fm"]  
                    #wfm = WFM.get_results()["wfm"]  
                    #sm = SM.get_results()["sm"]  
                    #em = EM.get_results()["em"]  
                    #mae = M.get_results()["mae"]  

                    # Guardar métricas de la imagen actual en el archivo  
                    #per_image_file.write(  
                    #    f"{base_name},{sm},{wfm},{mae},{em['adp']},{em['curve'].mean()},{em['curve'].max()},"  
                    #    f"{fm['adp']},{fm['curve'].mean()},{fm['curve'].max()}\n"  
                    #)  

            except Exception as e:  
                print(f"Error processing {base_name}: {str(e)}")  
                continue  

    # Calcular métricas globales  
    fm = FM.get_results()["fm"]  
    wfm = WFM.get_results()["wfm"]  
    sm = SM.get_results()["sm"]  
    em = EM.get_results()["em"]  
    mae = M.get_results()["mae"]  

    results = {  
        "Smeasure": sm,  
        "wFmeasure": wfm,  
        "MAE": mae,  
        "adpEm": em["adp"],  
        "meanEm": em["curve"].mean(),  
        "maxEm": em["curve"].max(),  
        "adpFm": fm["adp"],  
        "meanFm": fm["curve"].mean(),  
        "maxFm": fm["curve"].max(),  
    }  

    print(results)  
    with open("evalresults.txt", "a") as file:  
        file.write(f"{method} {_data_name} {str(results)}\n")  

"""        
import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

import cv2  
from tqdm import tqdm  
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure  

tecnique = 'BGNet-master'
method = 'BGNet_CottonWorm2_29'

tecnique = 'SINet-V2-main'
method = 'SINet_V2_CottonWorm'  

tecnique = 'PlantCamo-main'
method = 'CottonWorm1'  

tecnique = 'HitNet-main'
method = 'Hitnet_Cotton_Worm2'  

tecnique = 'DGNet-main'
method = 'CottonWorm2' 

tecnique = 'YOLOv8-main'
method = 'YOLOv8Net' 


def get_file_with_extension(directory, filename_without_ext):  
    #Busca un archivo con cualquier extensión en el directorio dado
    for ext in ['.png', '.jpg', '.jpeg']:  # Añade más extensiones si es necesario  
        potential_file = filename_without_ext + ext  
        if os.path.exists(os.path.join(directory, potential_file)):  
            return potential_file  
    return None  

def resize_to_match(img1, img2):  
    #Redimensiona img2 para que coincida con las dimensiones de img1
    if img1.shape != img2.shape:  
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)  
    return img2  
  
for _data_name in ['CottonWorm2']:  
    mask_root = f'C:/Respaldo/Henry/Proyecto Camuflaje/Datasets/{_data_name}/test/GT'  
    
    if tecnique == 'DGNet-main':
        pred_root = f'./{tecnique}/lib_pytorch/results/{method}/{_data_name}/'  
    else:
        pred_root = f'./{tecnique}/results/{method}/{_data_name}/'  

    # Obtener lista de archivos sin extensión  
    mask_names = [os.path.splitext(f)[0] for f in sorted(os.listdir(mask_root))]  

    FM = Fmeasure()  
    WFM = WeightedFmeasure()  
    SM = Smeasure()  
    EM = Emeasure()  
    M = MAE()  

    for base_name in tqdm(mask_names, total=len(mask_names)):  
        try:  
            # Encontrar los archivos con sus respectivas extensiones  
            mask_file = get_file_with_extension(mask_root, base_name)  
            pred_file = get_file_with_extension(pred_root, base_name)  

            if mask_file and pred_file:  
                mask_path = os.path.join(mask_root, mask_file)  
                pred_path = os.path.join(pred_root, pred_file)  

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  
                
                if mask is None or pred is None:  
                    print(f"Error: No se pudo leer la imagen {base_name}")  
                    continue  
                                    
                # Redimensionar pred para que coincida con mask  
                pred = resize_to_match(mask, pred)  
                  
                # Verificar que ambas imágenes tengan el mismo tamaño  
                if mask.shape != pred.shape:  
                    print(f"Error: Las dimensiones no coinciden después del resize para {base_name}")  
                    continue  
                # Imprimir dimensiones para debug  
                print(f"Original sizes - Mask: {mask.shape}, Pred: {pred.shape}")  

                FM.step(pred=pred, gt=mask)  
                WFM.step(pred=pred, gt=mask)  
                SM.step(pred=pred, gt=mask)  
                EM.step(pred=pred, gt=mask)  
                M.step(pred=pred, gt=mask)  
        except Exception as e:  
            print(f"Error processing {base_name}: {str(e)}")  
            continue  

    # Resto del código igual...  
    fm = FM.get_results()["fm"]  
    wfm = WFM.get_results()["wfm"]  
    sm = SM.get_results()["sm"]  
    em = EM.get_results()["em"]  
    mae = M.get_results()["mae"]  

    results = {  
        "Smeasure": sm,  
        "wFmeasure": wfm,  
        "MAE": mae,  
        "adpEm": em["adp"],  
        "meanEm": em["curve"].mean(),  
        "maxEm": em["curve"].max(),  
        "adpFm": fm["adp"],  
        "meanFm": fm["curve"].mean(),  
        "maxFm": fm["curve"].max(),  
    }  

    print(results)  
    with open("evalresults.txt", "a") as file:  
        file.write(f"{method} {_data_name} {str(results)}\n")  
        
"""