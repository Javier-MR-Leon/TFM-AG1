"""
PIPELINE ORQUESTADOR DE MACHINE LEARNING - TFM NEUROIMAGEN 
----------------------------------------------------------
Este script coordina el entrenamiento masivo y la auditoría final:
1. Ejecuta el Benchmarking (training.py).
2. Genera los Reportes Finales y Biomarcadores (evaluation.py).
"""

import os
import pandas as pd
from pathlib import Path

# Importamos las clases de tus archivos
from training import ModelBenchmarker
from evaluation import ModelEvaluator

def main():
    BASE_DIR = Path("./data/ESTUDIO_TFM")
    
    print("==================================================")
    print("   INICIANDO PIPELINE DE MACHINE LEARNING (ML)")
    print("==================================================")

    # Inicializar componentes
    benchmarker = ModelBenchmarker(str(BASE_DIR))
    evaluator = ModelEvaluator(str(BASE_DIR))

    # CARGA DE DATOS
    print("\n Cargando datasets finales...")
    datasets = benchmarker.cargar_datasets_maestros()
    
    if not datasets:
        print("❌ Error: No se encontraron datasets procesados en 3_FEATURES.")
        return

    # ENTRENAMIENTO (BENCHMARKING) 
    print("\n Ejecutando Benchmarking masivo (LOOCV)...")
    # Esto genera los archivos benchmarking_clasificacion.csv y benchmarking_regresion.csv
    df_res_clf, df_res_reg = benchmarker.ejecutar_benchmarking(datasets)

    # EVALUACIÓN Y BIOMARCADORES 
    print("\n Auditando modelos y extrayendo biomarcadores...")
    # Genera los reportes finales REPORTE_FINAL_... en 4_GRAPHICS
    df_maestra_clf, df_maestra_reg = evaluator.generar_reporte_final(
        df_res_clf, df_res_reg, datasets, umbral_overfit=15.0
    )

    print("\n Resumen de mejores modelos (Clasificación):")
    if not df_maestra_clf.empty:
        print(df_maestra_clf[['Dominio', 'Modalidad', 'Accuracy (%)', 'Modelo']].to_string(index=False))

    print("\n Resumen de mejores modelos (Regresión):")
    if not df_maestra_reg.empty:
        print(df_maestra_reg[['Dominio', 'Modalidad', 'MAE (Z)', 'Modelo']].to_string(index=False))

    print("\n==================================================")
    print("PROCESO DE ML FINALIZADO")
    print("==================================================")

if __name__ == "__main__":
    main()
