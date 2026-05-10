"""
MÓDULO DE ESTRUCTURACIÓN Y TRIAJE
---------------------------------
Este script prepara el árbol de directorios del proyecto y realiza un
filtro de calidad anatómico (mediante las dimensiones de la matriz) antes 
de copiar las imágenes a la carpeta de procesamiento.
"""

import os
import shutil
from pathlib import Path
import nibabel as nib

def preparar_entorno_y_explorar(base_path_str, estudio_path_str):
    """
    Prepara la estructura de directorios, realiza el triaje clínico 
    y aísla las imágenes viables en una carpeta de datos limpios.
    """
    
    # 1. Definición de rutas
    base_dir = Path(base_path_str)
    raw_dir = base_dir / "DATA_RAW/2_T1_CLEAN" # Carpeta donde terminaron los NIfTI del script anterior
    estudio_dir = Path(estudio_path_str)
    valid_dir = estudio_dir / "0_VALID_IMAGES"
    
    output_dirs = [
        valid_dir,
        estudio_dir / "1_SEGMENTATION",
        estudio_dir / "2_QC",
        estudio_dir / "3_FEATURES"
    ]
    
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f" Estructura creada en: {estudio_dir}")
        
    print("\n EXPLORANDO PACIENTES PARA TRIAJE")
    archivos_brutos = list(raw_dir.glob("*.nii")) + list(raw_dir.glob("*.nii.gz"))
    
    if not archivos_brutos:
        print(f" X No se encontraron archivos NIfTI en {raw_dir}")
        return []

    pacientes_procesados = {} 
    
    for archivo in archivos_brutos:
        # Extraemos el ID eliminando las extensiones
        nombre_paciente = archivo.name.replace('.nii.gz', '').replace('.nii', '')
        
        # Descartamos los archivos marcados como "prueba" o tests
        if "prueba" in nombre_paciente.lower():
            print(f" ---> Descartado (Etiqueta de prueba): {nombre_paciente}")
            continue
        
        # Evitamos duplicados
        if nombre_paciente in pacientes_procesados:
            continue
            
        try:
            # Verificamos la calidad de la matriz
            img = nib.load(archivo)
            dimensiones = img.shape
            
            # Se descartan aquellas imágenes < 50 cortes en caulquiera de los 3 ejes)
            if len(dimensiones) >= 3 and min(dimensiones[:3]) < 50:
                print(f" ---> Descartado (voxeles anisotrópicos con cortes gruesos): {nombre_paciente} | Matriz: {dimensiones}")

                archivo_malo_en_valid = valid_dir / archivo.name
                if archivo_malo_en_valid.exists():
                    archivo_malo_en_valid.unlink()
                continue
            
            pacientes_procesados[nombre_paciente] = {
                "id": nombre_paciente,
                "ruta_absoluta": archivo,
                "dimensiones": dimensiones
            }
            
            # Creamos subcarpeta individual para la posterior segmentación
            carpeta_paciente = estudio_dir / "1_SEGMENTATION" / nombre_paciente
            carpeta_paciente.mkdir(parents=True, exist_ok=True)
            
            # Copiamos el archivo validado a 0_VALID_IMAGES
            destino_nii = valid_dir / archivo.name
            if not destino_nii.exists():
                shutil.copy2(archivo, destino_nii)
            
            print(f" Validado y preparado: {nombre_paciente} | Matriz: {dimensiones}")
            
        except Exception as e:
            print(f" X Error procesando {archivo.name}: {e}")

    lista_final = list(pacientes_procesados.values())

    print(f"\n RESUMEN:")
    print(f"Total de sujetos viables para el estudio: {len(lista_final)}")
    print(f"Imágenes centralizadas en: {valid_dir}")
    
    return lista_final
