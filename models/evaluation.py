"""
MÓDULO DE EVALUACIÓN Y AUDITORÍA DE MODELOS
-------------------------------------------
Este script genera las tablas maestras finales para el TFM:
1. Selecciona el mejor modelo por dominio/modalidad (auditando sobreajuste).
2. Extrae la importancia de las variables.
3. Identifica los Top 5 Biomarcadores cerebrales (residuos puros).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.base import clone

try:
    from models.training import ModelBenchmarker
except ImportError:
    from training import ModelBenchmarker

class ModelEvaluator:
    def __init__(self, proyecto_dir):
        self.proyecto_dir = Path(proyecto_dir)
        self.graphics_dir = self.proyecto_dir / "4_GRAPHICS"
        self.benchmarker = ModelBenchmarker(proyecto_dir)

    def obtener_importancias(self, pipe, nombres_feat):
        """Extrae y normaliza la importancia de las variables del modelo entrenado."""
        selector = pipe.named_steps['selector']
        clf_final = pipe.named_steps['clf']
        
        # Obtener máscara de variables seleccionadas
        nombres_seleccionados = np.array(nombres_feat)[selector.get_support()].tolist()
        
        if hasattr(clf_final, 'coef_'):
            importancias = np.abs(clf_final.coef_[0] if len(clf_final.coef_.shape) > 1 else clf_final.coef_)
        elif hasattr(clf_final, 'feature_importances_'):
            importancias = clf_final.feature_importances_
        else:
            importancias = selector.scores_[selector.get_support()]

        importancias = np.atleast_1d(importancias).flatten()
        suma = np.sum(importancias)
        pct = (importancias / suma * 100) if suma > 0 else np.zeros(len(importancias))
        
        ranking = sorted(zip(nombres_seleccionados, pct), key=lambda x: x[1], reverse=True)[:5]
        return " | ".join([f"{n} ({p:.1f}%)" for n, p in ranking])

    def generar_reporte_final(self, df_res_clf, df_res_reg, datasets_dict, umbral_overfit=15.0):
        """Genera tablas maestras auditadas para clasificación y regresión."""
        print(f"\n Construyendo Reportes Auditados (Umbral Overfit: {umbral_overfit}%)...")
        
        clf_maestro = []
        reg_maestro = []
        configs_clf, configs_reg = self.benchmarker.obtener_configs()

        # PROCESAR CLASIFICACIÓN 
        grupos_clf = df_res_clf.groupby(['Dominio', 'Modalidad'])
        for (dom, mod), tabla in grupos_clf:
            # Ordenar por Accuracy y buscar el primero que no supere el umbral de Gap
            modelos = tabla.sort_values(by='Acc_Val (%)', ascending=False)
            
            for _, fila in modelos.iterrows():
                gap = fila['Gap_Sobreajuste (%)']
                nombre_mod = fila['Modelo']
                es_sobreajustado = gap > umbral_overfit
                
                # Re-entrenamiento para extracción de features
                df_orig = datasets_dict[mod].dropna(subset=[dom]).copy()
                targets = [c for c in df_orig.columns if c.startswith('D_')]
                cols_brain = df_orig.select_dtypes(include=[np.number]).drop(
                    columns=targets + ['id', 'Edad_RM', 'sexo'], errors='ignore'
                ).columns.tolist()
                
                df_adj = self.benchmarker.ajustar_por_residuos(df_orig, cols_brain)
                X, y = df_adj[cols_brain], (df_adj[dom] < -1.5).astype(int)
                
                conf = next((c for c in configs_clf if c['n'] == nombre_mod), None)
                if not conf: continue
                
                k_num = X.shape[1] if fila['Mejor_K'] == 'all' else int(fila['Mejor_K'])
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('selector', SelectKBest(score_func=f_classif, k=k_num)),
                    ('clf', clone(conf['m']))
                ])
                
                try:
                    pipe.fit(X, y)
                    top_5 = self.obtener_importancias(pipe, cols_brain)
                    
                    clf_maestro.append({
                        'Dominio': dom, 'Modalidad': mod, 
                        'Modelo': f"{nombre_mod} (OVERFIT)" if es_sobreajustado else nombre_mod,
                        'Accuracy (%)': fila['Acc_Val (%)'], 'Gap (%)': gap,
                        'Top 5 Biomarcadores (%)': top_5
                    })
                    if not es_sobreajustado: break # Nos quedamos con el mejor no sobreajustado
                except: continue

        # PROCESAR REGRESIÓN
        idx_ganadores = df_res_reg.groupby(['Dominio', 'Modalidad'])['MAE_Val (Z)'].idxmin()
        for _, fila in df_res_reg.loc[idx_ganadores].iterrows():
            dom, mod, nombre_mod = fila['Dominio'], fila['Modalidad'], fila['Modelo']
            
            df_orig = datasets_dict[mod].dropna(subset=[dom]).copy()
            targets = [c for c in df_orig.columns if c.startswith('D_')]
            cols_brain = df_orig.select_dtypes(include=[np.number]).drop(
                columns=targets + ['id', 'Edad_RM', 'sexo'], errors='ignore'
            ).columns.tolist()
            
            df_adj = self.benchmarker.ajustar_por_residuos(df_orig, cols_brain)
            X, y = df_adj[cols_brain], df_adj[dom]
            
            conf = next((c for c in configs_reg if c['n'] == nombre_mod), None)
            if not conf: continue
            
            k_num = X.shape[1] if fila['Mejor_K'] == 'all' else int(fila['Mejor_K'])
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(score_func=f_regression, k=k_num)),
                ('clf', clone(conf['m']))
            ])
            
            try:
                pipe.fit(X, y)
                top_5 = self.obtener_importancias(pipe, cols_brain)
                reg_maestro.append({
                    'Dominio': dom, 'Modalidad': mod, 'Modelo': nombre_mod,
                    'MAE (Z)': fila['MAE_Val (Z)'], 'Top 5 Predictores (%)': top_5
                })
            except: continue

        # Guardado final en 4_GRAPHICS
        df_clf_fin = pd.DataFrame(clf_maestro)
        df_reg_fin = pd.DataFrame(reg_maestro)
        
        df_clf_fin.to_csv(self.graphics_dir / "REPORTE_FINAL_CLASIFICACION.csv", index=False)
        df_reg_fin.to_csv(self.graphics_dir / "REPORTE_FINAL_REGRESION.csv", index=False)
        
        print(f" Reportes finales guardados en {self.graphics_dir}")
        return df_clf_fin, df_reg_fin
