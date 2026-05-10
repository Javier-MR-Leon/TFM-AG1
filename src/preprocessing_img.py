"""
MÓDULO DE PREPROCESAMIENTO DE LAS IMÁGENES DITCOM
-------------------------------------------------
Este script realiza el triaje técnico, la conversión de DICOM a NIfTI 
y la organización de secuencias T1 (priorizando 3D).
"""

import os
import re
import shutil
import unicodedata
from pathlib import Path
import nibabel as nib
import dicom2nifti
import pandas as pd
import pydicom

def limpiar_nombre(texto):
    """
    Asegura que el ID del paciente sea una cadena limpia.
    Elimina espacios y caracteres que puedan romper rutas en Docker.
    Idealmente el ID será un número.
    """
    # Convertimos a string por si llega un entero, quitamos espacios y caracteres raros
    texto = str(texto).strip()
    texto = re.sub(r'[^a-zA-Z0-9_]', '', texto) 
    return texto

def convertir_dicom_a_nifti(ruta_origen, ruta_destino, ruta_nifti):
    """
    Convierte carpetas DICOM a archivos NIfTI comprimidos (.nii.gz).
    Usa el nombre de la carpeta como ID del paciente para anonimización.
    También extrae la fecha en la que fue tomada la RM. 
    """
    ruta_origen = Path(ruta_origen)
    ruta_destino = Path(ruta_destino)
    ruta_destino.mkdir(parents=True, exist_ok=True)
    ruta_nifti = Path(ruta_nifti)
    ruta_nifti.mkdir(parents=True, exist_ok=True)

    print("\n INICIANDO CONVERSIÓN DE DICOM A NIFTI")
    registro_fechas = []

    for carpeta in ruta_origen.iterdir():
        # Si el elemento en la carpeta ya es un archivo NIfTI, lo copia directamente en la carpeta final.
        if carpeta.is_file() and (carpeta.name.endswith('.nii') or carpeta.name.endswith('.nii.gz')):
            nombre_final = f"ID_{limpiar_nombre(carpeta.name.split('.')[0])}.nii.gz"
            ruta_final = ruta_nifti / nombre_final
            
            if not ruta_final.exists():
                # Lo movemos/copiamos directamente a la carpeta final
                shutil.copy2(carpeta, ruta_final)
                print(f" NIfTI directo detectado y copiado: {nombre_final}")
            continue
      
        if carpeta.is_dir():
            # El ID será el nombre de la carpeta (número de historia asignado para este ejercicio)
            id_paciente = carpeta.name
            
            # Detectar estudios longitudinales (para carpetas que tengan subcarpetas numeradas)
            subcarpetas_fechas = [sub for sub in carpeta.iterdir() if sub.is_dir() and any(c.isdigit() for c in sub.name)]
            carpetas_a_procesar = subcarpetas_fechas if subcarpetas_fechas else [carpeta]

            for ruta_a_procesar in carpetas_a_procesar:
                nombre_final = f"ID_{id_paciente}_{ruta_a_procesar.name}" if subcarpetas_fechas else f"ID_{id_paciente}"
                
                # Extraer Fecha de la RM desde DICOM
                fecha_rm = "Desconocida"
                for archivo in ruta_a_procesar.rglob("*"):
                    if archivo.is_file():
                        try:
                            ds = pydicom.dcmread(archivo, stop_before_pixels=True)
                            if 'StudyDate' in ds:
                                f = ds.StudyDate
                                fecha_rm = f"{f[:4]}-{f[4:6]}-{f[6:]}"
                                break
                        except: continue

                # Evitar duplicados
                if (ruta_destino / f"{nombre_final}.nii.gz").exists():
                    print(f" --- Omitiendo {nombre_final}: Ya procesado.")
                    continue

                registro_fechas.append({'ID': nombre_final, 'Fecha_RM': fecha_rm})
                
                # Conversión
                try:
                    dest_temp = ruta_destino / nombre_final
                    dest_temp.mkdir(parents=True, exist_ok=True)
                    dicom2nifti.convert_directory(str(ruta_a_procesar), str(dest_temp), compression=True, reorient=True)

                    # Seguramente encontremos más de un archivo .nii por estudio (tendrán que ser revisados)
                    archivos_nii = list(dest_temp.glob("*.nii.gz"))
                    if len(archivos_nii) == 1:
                        shutil.move(str(archivos_nii[0]), str(ruta_destino / f"{nombre_final}.nii.gz"))
                        dest_temp.rmdir()
                        print(f" Convertido: {nombre_final}")
                    elif len(archivos_nii) > 1:
                        print(f" Múltiples secuencias en {nombre_final}. Revisar subcarpeta.")
                except Exception as e:
                    print(f" X Error en {nombre_final}: {e}")

    # Guardar registro de fechas
    if registro_fechas:
        pd.DataFrame(registro_fechas).to_csv(ruta_destino / "fechas_resonancias.csv", index=False)

def organizar_t1_final(ruta_entrada_str, ruta_salida_str):
    """
    Selecciona la mejor secuencia T1 (priorizando 3D) y la centraliza.
    """
    ruta_entrada = Path(ruta_entrada_str)
    ruta_salida = Path(ruta_salida_str)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    print(f"\n ORGANIZANDO ARCHIVOS T1 (PRIORIDAD 3D)")
    
    for carpeta_paciente in ruta_entrada.iterdir():
        if carpeta_paciente.is_dir():
            archivos_t1 = [f for f in carpeta_paciente.glob("*.nii*") if "t1" in f.name.lower()]
            if not archivos_t1: continue

            archivos_3d = [f for f in archivos_t1 if "3d" in f.name.lower()]
            archivo_elegido = archivos_3d[0] if archivos_3d else archivos_t1[0]
            indicador = "[3D]" if archivos_3d else "   [T1]"

            nuevo_nombre = f"{carpeta_paciente.name}_T1.nii.gz"
            try:
                shutil.copy2(archivo_elegido, ruta_salida / nuevo_nombre)
                print(f"{indicador} Copiado: {nuevo_nombre}")
            except Exception as e:
                print(f" X Error: {e}")

def auditar_resolucion(directorio_busqueda):
    """
    Analiza los archivos NIfTI y muestra su Voxel Size y dimensiones de matriz.
    Se usa como justifición de la exclusión de RMs con cortes gruesos.
    """
    archivos = list(Path(directorio_busqueda).rglob("*.nii*"))
    
    print(f"\n{'Paciente/Archivo':<40} | {'Voxel Size (mm)':<40} | {'Dimensiones (Matriz)':<20}")
    print("-" * 85)

    for ruta in archivos:
        try:
            img = nib.load(ruta)
            voxel_size = img.header.get_zooms()
            dimensiones = img.shape
            nombre_mostrable = ruta.name[:38]
            print(f"{nombre_mostrable:<40} | {str(voxel_size):<40} | {str(dimensiones):<20}")
        except Exception as e:
            print(f"Error en {ruta.name}: {e}")

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    # Rutas:
    PATH_RAW_DICOM = "./data/DATA_RAW/0_DICOM_RAW"
    PATH_CONVERTED = "./data/DATA_RAW/1_NIFTI_CONVERTED"
    PATH_FINAL_T1  = "./data/DATA_RAW/2_T1_CLEAN"

    # 1. Convertir DICOM a NIFTI
    convertir_dicom_a_nifti(PATH_RAW_DICOM, PATH_CONVERTED, PATH_FINAL_T1)

    # 2. Organizar T1 con prioridad 3D
    organizar_t1_final(PATH_CONVERTED, PATH_FINAL_T1)

    # 3. Auditar resolución técnica
    auditar_resolucion(PATH_FINAL_T1)

# RECURSOS: https://docs.python.org/es/3/library/pathlib.html
# https://pydicom.github.io/pydicom/stable/tutorials/dataset_basics.html
# https://pydicom.github.io/pydicom/dev/auto_examples/input_output/plot_read_dicom.html
# https://github.com/icometrix/dicom2nifti
# https://docs.python.org/es/3/library/shutil.html
