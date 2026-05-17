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
    bm.guardar_reportes_visuales(df_clf, df_reg)

    # Screening Mediación
    df_mediacion_crisis = benchmarker.screening_mediacion_crisis_total_tfm(dict_datasets=datasets, x_var='crisis_encefalop_ticas_v2')
    display(df_mediacion_crisis.head(20))
    
    df_mediacion_bioq = benchmarker.screening_mediacion_bioq_total_tfm(dict_datasets=datasets, x_var='bioquimico')
    display(df_mediacion_bioq.head(20))

    # EVALUACIÓN Y BIOMARCADORES 
    print("\n Auditando modelos y extrayendo relaciones...")
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

    # Se pintan los cerebros 3D y subcorticales en lote
    evaluador.ejecutar_renderizado_lote_neuroimagen(df_clf_fin, df_reg_fin)

    print(f"Reporte top mediadores para Bioquímico")
    evaluador.generar_reporte_top_mediadores_v7(df_mega_res_bioq, datasets_listos, x_var='bioquimico', p_umbral=0.15)

    print(f"\nReporte top mediadores para Crisis")
    evaluador.generar_reporte_top_mediadores_v7(df_mega_res_crisis, datasets_listos, x_var='crisis_encefalop_ticas_v2', p_umbral=0.15)

    print("\n==================================================")
    print("PROCESO DE ML Y MEDIACIÓN FINALIZADO")
    print("==================================================")

if __name__ == "__main__":
    main()
