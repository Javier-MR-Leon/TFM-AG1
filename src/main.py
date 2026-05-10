"""
PIPELINE TRATAMIENTO DE LOS DATOS - TFM NEUROIMAGEN
---------------------------------------------------
Orquestador principal que conecta los módulos de:
1. Preprocesamiento de Imagen (DICOM a NIfTI)
2. Estructuración y Triaje (Filtro de calidad)
3. Segmentación y Extracción (Docker SynthSeg/FastSurfer + Radiómica)
4. Feature Engineering (Normalización, EDA y Poda)
"""

import os
from pathlib import Path

# Importación de los módulos locales
from preprocessing_img import convertir_dicom_a_nifti, organizar_t1_final, auditar_resolucion
from structure import preparar_entorno_y_explorar
from segmentation_features import (
    ejecutar_synthseg_lote_docker, 
    ejecutar_fastsurfer_lote_docker, 
    extraer_radiomica_lote, 
    extraer_grosor_cortical_completo
)
from preprocessing_feat import FeatureProcessor

def main():
    # --- 0. CONFIGURACIÓN DE RUTAS ---
    BASE_DIR = Path("./data")
    RUTA_ESTUDIO = BASE_DIR / "ESTUDIO_TFM"
    RUTA_RAW_DICOM = BASE_DIR / "DATA_RAW/0_DICOM_RAW"
    RUTA_CONVERTED = BASE_DIR / "DATA_RAW/1_NIFTI_CONVERTED"
    RUTA_T1_CLEAN = BASE_DIR / "DATA_RAW/2_T1_CLEAN"
    RUTA_EXCEL_CLINICO = BASE_DIR / "DATA_RAW/clinical_data.xlsx"
    FS_LICENSE = BASE_DIR / "LICENSE_FS/license.txt"

    print("==================================================")
    print("   INICIANDO PIPELINE COMPLETO DE NEUROIMAGEN")
    print("==================================================")

    # --- 1. MÓDULO: PREPROCESAMIENTO DE IMAGEN ---
    # Convierte DICOM a NIfTI y selecciona la mejor T1 (3D)
    convertir_dicom_a_nifti(RUTA_RAW_DICOM, RUTA_CONVERTED, RUTA_T1_CLEAN)
    organizar_t1_final(RUTA_CONVERTED, RUTA_T1_CLEAN)
    auditar_resolucion(RUTA_T1_CLEAN)

    # --- 2. MÓDULO: ESTRUCTURACIÓN Y TRIAJE ---
    # Crea carpetas y filtra imágenes por dimensiones de matriz (Triaje Técnico)
    # Devuelve la lista de pacientes que pasaron el filtro de calidad
    lista_pacientes = preparar_entorno_y_explorar(str(BASE_DIR), str(RUTA_ESTUDIO))

    # --- 3. MÓDULO: SEGMENTACIÓN Y EXTRACCIÓN ---
    # Orquestación de contenedores Docker y algoritmos de extracción
    print("\nLanzando procesos pesados (Docker)...")
    
    # Segmentación Subcortical (SynthSeg)
    ejecutar_synthseg_lote_docker(lista_pacientes, str(RUTA_ESTUDIO))
    
    # Morfometría de Superficie (FastSurfer)
    ejecutar_fastsurfer_lote_docker(lista_pacientes, str(RUTA_ESTUDIO), str(FS_LICENSE))
    
    # Extracción de Radiómica (Ganglios Basales)
    extraer_radiomica_lote(lista_pacientes, str(RUTA_ESTUDIO))
    
    # Extracción de Grosor Cortical (Atlas DKT)
    extraer_grosor_cortical_completo(lista_pacientes, str(RUTA_ESTUDIO))

    # --- 4. MÓDULO: FEATURE ENGINEERING ---
    # Normalización, Vinculación clínica, EDA y Poda de Colinealidad
    processor = FeatureProcessor(str(RUTA_ESTUDIO))
    
    # Definición de rutas para el procesador de features
    rutas_features = {
        "vol": RUTA_ESTUDIO / "3_FEATURES/volumenes_synthseg_todos.csv",
        "grosor": RUTA_ESTUDIO / "3_FEATURES/grosor_cortical_completo.csv",
        "radio": RUTA_ESTUDIO / "3_FEATURES/radiomica_results.csv"
    }

    # A. Limpieza de datos clínicos y creación de dominios D_
    df_clinico = processor.crear_dominios_clinicos(RUTA_EXCEL_CLINICO)
    ruta_clin_final = RUTA_ESTUDIO / "3_FEATURES/clinico_FINAL.csv"
    df_clinico.to_csv(ruta_clin_final, index=False)
    rutas_features["clinico"] = ruta_clin_final

    # B. Sanity Check Maestro
    processor.ejecutar_sanity_check_maestro(rutas_features, processor.qc_dir)

    # C. Normalización Multimodal (ICV y Z-Score)
    datasets_norm = processor.ejecutar_normalizacion_y_limpieza(rutas_features)

    # D. Vinculación de Metadatos y Targets
    datasets_vinc = processor.vincular_metadatos_y_targets(datasets_norm)

    # E. Análisis Exploratorio (EDA) - Genera gráficas en 4_GRAPHICS
    processor.ejecutar_eda_completo(datasets_vinc)

    # F. Poda Final de Colinealidad (Datasets listos para ML)
    processor.ejecutar_poda_multimodal(datasets_vinc)

    print("\n==================================================")
    print("PIPELINE FINALIZADO CON ÉXITO")
    print("==================================================")

if __name__ == "__main__":
    main()
