"""
MÓDULO DE SEGMENTACIÓN Y EXTRACCIÓN DE CARACTERÍSTICAS
-----------------------------------------------------
Este script orquesta contenedores Docker (SynthSeg y FastSurfer) para 
la segmentación neuroanatómica y extrae métricas de Radiómica y 
Grosor Cortical.
"""

import os
import time
import subprocess
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from radiomics import featureextractor

def ejecutar_synthseg_lote_docker(lista_pacientes, estudio_dir_str):
    """
    Orquesta SynthSeg (FreeSurfer) mediante Docker para segmentación subcortical.
    Genera volúmenes y máscaras, unificando los resultados en un CSV maestro.
    """
    estudio_dir = Path(estudio_dir_str)
    print("\n INICIANDO PIPELINE DE SEGMENTACIÓN (SYNTHSEG)")
    tiempo_inicio_total = time.time()
    rutas_csv_individuales = []
    
    barra_progreso = tqdm(lista_pacientes, desc="Segmentando cerebros", unit="paciente")
    
    for paciente in barra_progreso:
        id_paciente = paciente['id']
        nombre_archivo = paciente['ruta_absolute'].name
        barra_progreso.set_description(f"Procesando: {id_paciente}")
        
        carpeta_salida = estudio_dir / "1_SEGMENTATION" / id_paciente
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        
        # Rutas mapeadas al contenedor Docker
        ruta_input_docker = f"./data/ESTUDIO_TFM/0_VALID_IMAGES/{nombre_archivo}"
        ruta_output_docker = f"./data/ESTUDIO_TFM/1_SEGMENTATION/{id_paciente}/{id_paciente}_synthseg.nii.gz"
        ruta_vol_docker = f"./data/ESTUDIO_TFM/1_SEGMENTATION/{id_paciente}/{id_paciente}_volumenes.csv"
        archivo_vol_win = carpeta_salida / f"{id_paciente}_volumenes.csv"
        
        if archivo_vol_win.exists():
            rutas_csv_individuales.append(archivo_vol_win)
            continue
            
        comando = [
            "docker", "run", "--rm", 
            "-v", f"{estudio_dir_str}:/data", 
            "freesurfer/freesurfer:7.4.1", 
            "mri_synthseg", "--i", ruta_input_docker, 
            "--o", ruta_output_docker, "--vol", ruta_vol_docker
        ]
        
        try:
            inicio_paciente = time.time()
            
            subprocess.run(comando, check=True, capture_output=True, text=True)
            
            fin_paciente = time.time()
            minutos_paciente = (fin_paciente - inicio_paciente) / 60
            tqdm.write(f" ---> {id_paciente} completado en {minutos_paciente:.1f} min.")
            
            if archivo_vol_win.exists():
                rutas_csv_individuales.append(archivo_vol_win)
        except subprocess.CalledProcessError as e:
            tqdm.write(f" X Error en {id_paciente}: {e.stderr}")

    # Unificación de datos volumétricos
    if rutas_csv_individuales:
        print("\n Unificando volúmenes en CSV maestro...")
        df_list = []
        for csv_path in rutas_csv_individuales:
            df = pd.read_csv(csv_path)
            df.insert(0, 'id_paciente_limpio', csv_path.stem.replace('_volumenes', ''))
            df_list.append(df)
        
        df_conjunto = pd.concat(df_list, ignore_index=True)
        df_conjunto.to_csv(estudio_dir / "3_FEATURES" / "volumenes_synthseg_todos.csv", index=False)

    tiempo_fin_total = time.time()
    horas_totales = (tiempo_fin_total - tiempo_inicio_total) / 3600
    print(f"\n SEGMENTACIÓN FINALIZADA. Tiempo total de procesamiento: {horas_totales:.2f} horas.")

def ejecutar_fastsurfer_lote_docker(lista_pacientes, estudio_dir_str, fs_license_path):
    """
    Ejecuta FastSurfer para morfometría de superficie. 
    Optimizado para Windows mediante el uso de CPU en la agregación de vistas
    para evitar desbordamientos de memoria.
    """
    estudio_dir = Path(estudio_dir_str)
    tiempo_inicio_total = time.time()
    print("\n INICIANDO PIPELINE MORFOMETRÍA DE SUPERFICIE (FASTSURFER)")
    
    
    for paciente in tqdm(lista_pacientes, desc="Procesando Superficie"):
        id_p = paciente['id']
        t1_file = paciente['ruta_absolute'].name
        out_dir = estudio_dir / "1_SEGMENTATION" / "1_FASTSURFER_OUT"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            check_file = sd_dir / id_paciente / "surf" / "lh.thickness"
            if check_file.exists():
                tqdm.write(f"---> {id_paciente} ya tiene resultados. Saltando...")
                continue

        comando = [
            "docker", "run", "--rm", "--gpus", "all", "--user", "root",
            "-v", f"{estudio_dir_str}:/data",
            "-v", f"{fs_license_path}:/opt/freesurfer/license.txt",
            "--env", "FS_LICENSE=/opt/freesurfer/license.txt", 
            "deepmi/fastsurfer:latest", "--allow_root",
            "--t1", f"/data/0_VALID_IMAGES/{t1_file}",
            "--sd", "/data/1_SEGMENTATION/1_FASTSURFER_OUT",
            "--sid", id_p, "--vox_size", "1.0", "--viewagg_device", "cpu"
        ]

        try:
            subprocess.run(comando, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        except subprocess.CalledProcessError as e:
            tqdm.write(f" X Error FastSurfer en {id_p}: {e.stderr}")

      tiempo_fin_total = time.time()
      horas_totales = (tiempo_fin_total - tiempo_inicio_total) / 3600
      print(f"\n MORFOMETRÍA DE SUPERFICIE FINALIZADA. Tiempo total de procesamiento: {horas_totales:.2f} horas.")
      
      if pacientes_fallidos:
        print(f"\n X El proceso terminó con errores en estos pacientes: {pacientes_fallidos}")
        print("Revisa las imágenes originales de estos sujetos; pueden tener mucho ruido o artefactos.")

def extraer_radiomica_lote(lista_pacientes, estudio_dir):
    """
    Extrae características de textura mediante PyRadiomics.
    Ajusta previamente la geometría del T1.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['geometryTolerance'] = 1e-3
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')

    resultados = []
    labels = {'Putamen_L': 12, 'Putamen_R': 51, 'Caudado_L': 11, 'Caudado_R': 50, 'Palido_L': 13, 'Palido_R': 52, 'Talamo_L': 10, 'Talamo_R': 49,
              'Accumbens_L': 26, 'Accumbens_R': 58, 'Sust_Blanca_L': 2, 'Sust_Blanca_R': 41, 'Tronco_Encefalico': 16, 'Ventriculos_Lat_L': 4, 
              'Ventriculos_Lat_R': 43, 'LCR_Extracerebral': 24}

    for p in lista_pacientes:
        p_id = p['id']
        t1_path = str(p['ruta_absolute'])
        seg_path = os.path.join(estudio_dir, "1_SEGMENTATION", p_id, f"{p_id}_synthseg.nii.gz")
        t1_adj_path = os.path.join(estudio_dir, "1_SEGMENTATION", p_id, f"{p_id}_t1_ajustado.nii.gz")

        if not os.path.exists(seg_path): continue

        # Alineación geométrica mediante SimpleITK
        if not os.path.exists(t1_adj_path):
            t1_img = sitk.ReadImage(t1_path)
            mask_img = sitk.ReadImage(seg_path)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(mask_img)
            resampler.SetInterpolator(sitk.sitkLinear)
            sitk.WriteImage(resampler.Execute(t1_img), t1_adj_path)

        # Extracción por ROI
        datos = {'ID': p_id}
        for roi, val in labels.items():
            try:
                f = extractor.execute(t1_adj_path, seg_path, label=val)
                datos[f'{roi}_Intensity'] = f.get('diagnostics_Image-original_Mean')
                datos[f'{roi}_Entropy'] = f.get('original_firstorder_Entropy')
            except: continue
        resultados.append(datos)

    pd.DataFrame(resultados).to_csv(os.path.join(estudio_dir, "3_FEATURES", "radiomica_results.csv"), index=False)

def extraer_grosor_cortical_completo(lista_pacientes, estudio_dir):
    """
    Parsea los archivos de estadísticas de FastSurfer (.stats) para 
    extraer el grosor medio de todas las regiones del Atlas DKT.
    """
    fs_dir = os.path.join(estudio_dir, "1_SEGMENTATION", "1_FASTSURFER_OUT")
    resultados = []

    print("\n EXTRAYENDO EL GROSOR CORTICAL COMPLETO (ATLAS DKT) ")

    for p in lista_pacientes:
        p_id = p['id']
        datos = {'ID': p_id}
        for hemi in ['lh', 'rh']:
            path = os.path.join(fs_dir, p_id, "stats", f"{hemi}.aparc.DKTatlas.mapped.stats")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for line in f:
                        if line.startswith('#'): continue
                        parts = line.split()
                        if len(parts) > 5:
                            datos[f'{hemi}_{parts[0]}_thick'] = float(parts[4])
        resultados.append(datos)

    pd.DataFrame(resultados).to_csv(os.path.join(estudio_dir, "3_FEATURES", "grosor_cortical.csv"), index=False)
    print(f"\n GROSOR CORTICAL EXTRAIDO")

# RECURSOS: https://www.datacamp.com/es/tutorial/tqdm-python
# https://docs.python.org/es/3/library/subprocess.html
# https://surfer.nmr.mgh.harvard.edu/fswiki/infantFS-containers
# https://stackoverflow.com/questions/79009052/run-docker-command-within-python-subprocess-within-github-runner
# https://www.pythonsnacks.com/p/python-subprocess-shell-command 
# https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all-clinical
# https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg
# https://github.com/BBillot/SynthSeg
# https://pyradiomics.readthedocs.io/en/latest/radiomics.html
# https://simpleitk.readthedocs.io/en/master/link_DemonsRegistration1_docs.html
# 
