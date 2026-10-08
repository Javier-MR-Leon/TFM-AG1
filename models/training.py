"""
MÓDULO DE MACHINE LEARNING - TRAINING & BENCHMARKING
----------------------------------------------------
Este script realiza el entrenamiento masivo buscando el mejor modelo 
para cada dominio neurocognitivo.
"""

import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# Scikit-learn Base
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB

# Modelos (Clasificación y Regresión)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, Ridge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class ModelBenchmarker:
    def __init__(self, proyecto_dir):
        self.proyecto_dir = Path(proyecto_dir)
        self.features_dir = self.proyecto_dir / "3_FEATURES"
        self.graphics_dir = self.proyecto_dir / "4_GRAPHICS"
        self.clean_dir = self.features_dir / "DATA_CLEAN"

    def cargar_datasets_maestros(self):
        """Busca y carga los archivos dataset_ML_*_FINAL.csv y el clínico."""
        datasets = {}
        archivos = list(self.features_dir.glob("dataset_ML_*_FINAL.csv"))
        
        if not archivos:
            print(f" No se encontraron archivos finales en {self.features_dir}")
            return datasets

        for arc in archivos:
            nombre = arc.stem.replace("dataset_ML_", "").replace("_FINAL", "")
            df = pd.read_csv(arc)
            if 'id' in df.columns: df['id'] = df['id'].astype(str)
            if 'ID' in df.columns: df['ID'] = df['ID'].astype(str)
            cols_num = df.select_dtypes(include=[np.number]).columns
            df[cols_num] = df[cols_num].astype(float)
            datasets[nombre] = df

        # Cargar específicamente el archivo clínico externo que mencionas
        path_clinico = self.features_dir / "clinico_FINAL_NORM.csv"
        if path_clinico.exists():
            df_clin = pd.read_csv(path_clinico)
            # Normalizar nombre de la columna ID si varía
            for col_id in ['id', 'ID', 'record_id']:
                if col_id in df_clin.columns:
                    df_clin['ID'] = df_clin[col_id].astype(str).str.strip()
                    break
            datasets['clinico'] = df_clin
            print(f"Dataset clínico externo cargado con éxito desde: {path_clinico.name}")
        else:
            print(f"X No se encontró el archivo clínico en: {path_clinico}")

        return datasets

    def ajustar_por_residuos(self, df, columnas_cerebrales):
        """Elimina la varianza explicada por Edad y Sexo (Ajuste multivariable)."""
        df_ajustado = df.copy()
        lr = LinearRegression()
        covs_f = ['Edad_RM', 'sexo']

        for col in columnas_cerebrales:
            mask = df_ajustado[[col] + covs_f].notna().all(axis=1)
            if mask.sum() < 15: continue
            
            X = df_ajustado.loc[mask, covs_f].values
            y = df_ajustado.loc[mask, col].values
            lr.fit(X, y)
            df_ajustado.loc[mask, col] = y - lr.predict(X)
            
        return df_ajustado

    def obtener_configs(self):
        """Define los diccionarios de modelos y parámetros del código original."""
        clf = [
            {'n': 'LASSO (L1)', 'm': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42),
             'p': {'selector__k': [5, 10, 15, 'all'], 'clf__C': [0.01, 0.1, 1, 10, 100]}},
            {'n': 'Ridge (L2)', 'm': RidgeClassifier(class_weight='balanced', random_state=42),
             'p': {'selector__k': [5, 10, 15, 'all'], 'clf__alpha': [0.01, 0.1, 1, 10, 100]}},
            {'n': 'Elastic Net', 'm': LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', random_state=42, max_iter=5000),
             'p': {'selector__k': [5, 10, 15, 'all'], 'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__l1_ratio': [0.2, 0.5, 0.7, 0.9]}},
            {'n': 'SVM Lineal', 'm': SVC(kernel='linear', class_weight='balanced', random_state=42),
             'p': {'selector__k': [5, 10, 15, 'all'], 'clf__C': [0.001, 0.01, 0.1, 1, 10]}},
            {'n': 'Árbol Decisión', 'm': DecisionTreeClassifier(class_weight='balanced', random_state=42),
             'p': {'selector__k': [5, 10, 15], 'clf__max_depth': [2, 3, 4]}},
            {'n': 'KNN', 'm': KNeighborsClassifier(),
             'p': {'selector__k': [5, 10], 'clf__n_neighbors': [3, 5], 'clf__weights': ['uniform', 'distance']}},
            {'n': 'Naive Bayes', 'm': BernoulliNB(),
             'p': {'selector__k': [5, 10], 'clf__alpha': [0.1, 1.0]}}
        ]
        reg = [
            {'n': 'Lasso Reg', 'm': Lasso(random_state=42),
             'p': {'selector__k': [5, 10, 15, 'all'], 'clf__alpha': [0.01, 0.1, 1, 10]}},
            {'n': 'Ridge Reg', 'm': Ridge(random_state=42),
             'p': {'selector__k': [5, 10, 15, 'all'], 'clf__alpha': [0.1, 1, 10, 100]}},
            {'n': 'ElasticNet Reg', 'm': ElasticNet(random_state=42),
             'p': {'selector__k': [5, 10, 15, 'all'], 'clf__alpha': [0.01, 0.1, 1], 'clf__l1_ratio': [0.2, 0.5, 0.8]}},
            {'n': 'SVR Lineal', 'm': SVR(kernel='linear'),
             'p': {'selector__k': [5, 10, 15, 'all'], 'clf__C': [0.1, 1, 10], 'clf__epsilon': [0.01, 0.1]}},
            {'n': 'Árbol Regresión', 'm': DecisionTreeRegressor(random_state=42),
             'p': {'selector__k': [5, 10, 15], 'clf__max_depth': [2, 3, 4]}},
            {'n': 'KNN Regressor', 'm': KNeighborsRegressor(),
             'p': {'selector__k': [5, 10], 'clf__n_neighbors': [3, 5], 'clf__weights': ['uniform', 'distance']}}
        ]
        return clf, reg


    def ejecutar_benchmarking(self, dict_datasets):
        print("\n[INICIANDO BENCHMARKING]")
        res_clf, res_reg = [], []
        loo = LeaveOneOut()
        conf_clf, conf_reg = self.obtener_configs()

        for nombre_mod, df in dict_datasets.items():
            print(f"\nDATASET: {nombre_mod.upper()} (Filas originales: {len(df)})")

            targets = [c for c in df.columns if c.startswith('D_')]
            no_tocar = targets + ['ID', 'Edad_RM', 'sexo']
            features = [c for c in df.columns if c not in no_tocar and df[c].dtype in ['float64', 'int64']]
            
            if not features:
                continue

            # 1. Auditoría transparente de NaNs por columna en las features
            df_features_sub = df[features]
            nan_counts = df_features_sub.isna().sum()
            cols_con_nan = nan_counts[nan_counts > 0]
            
            if not cols_con_nan.empty:
                print(f"X Columnas con valores nulos ({len(cols_con_nan)}):")
                for col, count in cols_con_nan.items():
                    pct = (count / len(df)) * 100
                    print(f"      - {col}: {count} nulos ({pct:.1f}%)")
                
                # Mostrar qué IDs exactos tienen nulos en las features
                filas_con_nulos = df[df[features].isna().any(axis=1)]
                if 'ID' in filas_con_nulos.columns:
                    print(f"\n ---> IDs de sujetos afectados: {filas_con_nulos['ID'].tolist()}")
            else:
                print("\n¡Cero valores nulos en las features de este dataset!")

            # 2. Exclusión estricta de filas con nulos (Sin imputación)
            df_clean = df.dropna(subset=features).copy()
            if len(df_clean) < len(df):
                print(f"\n -> Excluyendo rigurosamente {len(df) - len(df_clean)} filas con NaNs.")

            X = df_clean[features].values

            for target in targets:
                y = df_clean[target].values
                
                # Filtrar filas donde el target sea NaN
                val_mask = ~np.isnan(y)
                X_t = X[val_mask]
                y_t = y[val_mask]
                
                if len(y_t) < 5:
                    continue

                # --- CLASIFICACIÓN ---
                y_clf = np.where(y_t < -1.5, 1, 0) # 1: Déficit, 0: Normal
                if len(np.unique(y_clf)) > 1:
                    print(f"   Optimizando CLF para {target} en {nombre_mod}...")
                    for conf in conf_clf:
                        try:
                            # Pipeline clásico sin imputadores, operando sobre datos limpios
                            pipe = Pipeline([
                                ('scaler', StandardScaler()), 
                                ('selector', SelectKBest(f_classif)), 
                                ('clf', conf['m'])
                            ])
                            grid = GridSearchCV(pipe, conf['p'], cv=loo, scoring='accuracy', n_jobs=-1, return_train_score=True)
                            grid.fit(X_t, y_clf)
                            res_clf.append({
                                'Modalidad': nombre_mod, 'Dominio': target, 'Modelo': conf['n'],
                                'Acc_Val (%)': round(grid.best_score_ * 100, 2),
                                'Gap_Sobreajuste (%)': round((grid.cv_results_['mean_train_score'][grid.best_index_] - grid.best_score_) * 100, 2),
                                'Mejor_K': grid.best_params_.get('selector__k')
                            })
                        except Exception as e:
                            continue

                # --- REGRESIÓN ---
                print(f"   Optimizando REG para {target} en {nombre_mod}...")
                for conf in conf_reg:
                    try:
                        pipe = Pipeline([
                            ('scaler', StandardScaler()), 
                            ('selector', SelectKBest(f_regression)), 
                            ('clf', conf['m'])
                        ])
                        grid = GridSearchCV(pipe, conf['p'], cv=loo, scoring='neg_mean_squared_error', n_jobs=-1, return_train_score=True)
                        grid.fit(X_t, y_t)
                        mse_val = -grid.best_score_
                        res_reg.append({
                            'Modalidad': nombre_mod, 'Dominio': target, 'Modelo': conf['n'],
                            'MSE_Val': round(mse_val, 4),
                            'Mejor_K': grid.best_params_.get('selector__k')
                        })
                    except Exception as e:
                        continue

        df_clf = pd.DataFrame(res_clf)
        df_reg = pd.DataFrame(res_reg)
        return df_clf, df_reg

    def guardar_reportes_visuales(self, df_clf, df_reg):
        """Genera mapas de calor con métricas y sobreajuste."""
        sns.set_theme(style="white")
        
        # CLASIFICACIÓN (Accuracy + Sobreajuste en el texto) 
        for m in df_clf['Modelo'].unique():
            df_m = df_clf[df_clf['Modelo'] == m]
            
            # Pivotar datos
            tabla_acc = df_m.pivot(index='Dominio', columns='Modalidad', values='Acc_Val (%)')
            tabla_gap = df_m.pivot(index='Dominio', columns='Modalidad', values='Gap_Sobreajuste (%)')

            # Crear etiquetas personalizadas: "Acc% (Gap%)"
            etiquetas = np.array([[f"{acc:.1f}\n({gap:+.1f})" for acc, gap in zip(row_acc, row_gap)] 
                                 for row_acc, row_gap in zip(tabla_acc.values, tabla_gap.values)])

            plt.figure(figsize=(11, 7))
            sns.heatmap(tabla_acc, annot=etiquetas, fmt="", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
            plt.title(f"Benchmarking Clasificación: {m}\nAccuracy % (Sobreajuste % entre paréntesis)", fontsize=14, pad=20)
            
            nombre_arch = f"RESULTADOS_CLF_{m.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plt.savefig(self.graphics_dir / nombre_arch, dpi=300, bbox_inches='tight')
            plt.close()

        # REGRESIÓN (MAE) 
        for m in df_reg['Modelo'].unique():
            df_m = df_reg[df_reg['Modelo'] == m]
            tabla_mae = df_m.pivot(index='Dominio', columns='Modalidad', values='MSE_Val')
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(tabla_mae, annot=True, cmap="RdYlGn_r", fmt=".3f", cbar_kws={'label': 'MAE (Error en Z)'})
            plt.title(f"Benchmarking Regresión: {m}\n(Error MAE en Puntuación Z)", fontsize=14, pad=20)
            
            nombre_arch = f"RESULTADOS_REG_{m.replace(' ', '_')}.png"
            plt.savefig(self.graphics_dir / nombre_arch, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\n IMÁGENES GUARDADAS en: {self.graphics_dir}")

    def screening_mediacion_crisis_total_tfm(self, dict_datasets, x_var='crisis_encefalop_ticas_v2'):
        """X: Variable clínica | M: Mediadores de Imagen | Y: Dominios Cognitivos"""
        print(f"\n --- INICIANDO MEGA-SCREENING MULTITAREA (X = {x_var}) ---")

        df_clin = dict_datasets["clinico"].copy()
        
        # Auditoría previa: Ver tipo de dato y muestra cruda antes de limpiar
        print(f" [Auditoría] Tipo de ID en clínico: {df_clin['ID'].dtype} | Muestra: {df_clin['ID'].head(5).tolist()}")
        
        # Limpieza robusta: Quitar .0 de floats, cortar por '_' y limpiar espacios
        def limpiar_id_serie(s):
            return s.astype(str).str.split('_').str[0].str.replace(r'\.0$', '', regex=True).str.strip()

        df_clin['ID_clean'] = limpiar_id_serie(df_clin['ID'])
        targets_y = [c for c in df_clin.columns if c.startswith('D_')]

        resultados_globales = []
        datasets_analizar = ["vol", "morfometria", "radio"]
        
        for tipo_ds in datasets_analizar:
            print(f"\nProcesando Dominio Imagen: {tipo_ds}...")
            df_brain = dict_datasets[tipo_ds].copy()
            
            # Auditoría previa del dataset de imagen
            print(f" [Auditoría] Tipo de ID en {tipo_ds}: {df_brain['ID'].dtype} | Muestra: {df_brain['ID'].head(5).tolist()}")
            
            df_brain['ID_clean'] = limpiar_id_serie(df_brain['ID'])
            
            cols_drop = [c for c in df_brain.columns if c.startswith('D_') or c in ['sexo', 'Edad_RM']]
            df_brain_clean = df_brain.drop(columns=[c for c in cols_drop if c != 'ID' and c != 'ID_clean'])
            
            # Auditoría de intersección: ver cuántos IDs coinciden exactamente antes del merge
            ids_brain = set(df_brain_clean['ID_clean'])
            ids_clin = set(df_clin['ID_clean'])
            coincidencias = ids_brain.intersection(ids_clin)
            print(f" IDs únicos en {tipo_ds}: {len(ids_brain)} | IDs únicos en clínico: {len(ids_clin)} | Coincidencias teóricas: {len(coincidencias)}")
            print(f"   -> IDs coincidentes: {sorted(list(coincidencias))}")

            # Merge seguro usando la columna limpia ID_clean y permitiendo left join para retener todo el cerebro
            df_master = pd.merge(df_brain_clean, df_clin, on='ID_clean', how='left', suffixes=('', '_clin'))
            print(f" Merge completado. Filas resultantes: {len(df_master)}")

            # Mapeo de X a numérico si es texto (Vital para Pingouin)
            if x_var in df_master.columns:
                if df_master[x_var].dtype == 'object' or pd.api.types.is_categorical_dtype(df_master[x_var]):
                    mapeo = {val: i for i, val in enumerate(df_master[x_var].dropna().unique())}
                    df_master['X_num'] = df_master[x_var].map(mapeo)
                else:
                    df_master['X_num'] = df_master[x_var]
            else:
                df_master['X_num'] = df_master.get(x_var, 0)

            mediadores = [c for c in df_brain_clean.columns if c not in ['ID', 'ID_clean', 'sexo', 'Edad_RM', x_var, 'crisis_encefalop_ticas_v2', 'bioquimico'] and not c.startswith('D_')]

            for target in targets_y:
                if target not in df_master.columns: continue
                for m_roi in mediadores:
                    if m_roi not in df_master.columns: continue
                    try:
                        df_model = df_master[['X_num', m_roi, target, 'sexo', 'Edad_RM']].dropna()
                        if len(df_model) < 5: continue # Umbral ajustado para n=14
                        
                        df_res = self.ajustar_por_residuos(df_model, [m_roi])
                        res = pg.mediation_analysis(data=df_res, x='X_num', m=m_roi, y=target, n_boot=200, seed=42)
                        
                        col_c = [c for c in res.columns if 'coef' in c.lower()][0]
                        col_p = [c for c in res.columns if 'pval' in c.lower() or 'p-val' in c.lower()][0]
                        
                        row_a = res[res['path'].str.contains('-> M|X', case=False, na=False)]
                        row_b = res[res['path'].str.contains('M ->|Y', case=False, na=False)]
                        row_ind = res[res['path'].str.contains('Indirect', case=False, na=False)]
                        
                        resultados_globales.append({
                            'Target_Y': target, 'Modalidad_M': tipo_ds, 'Mediador_M': m_roi,
                            'Coef_Indirecto': row_ind[col_c].values[0], 'P_val_Indirecto': row_ind[col_p].values[0],
                            'Sig': row_ind['sig'].values[0] if 'sig' in row_ind.columns else 'No',
                            'P_Path_A (Crisis->Medida)': row_a[col_p].values[0] if not row_a.empty else np.nan,
                            'P_Path_B (Medida->Dominio)': row_b[col_p].values[0] if not row_b.empty else np.nan
                        })
                    except: continue

        if not resultados_globales:
            print("X No se obtuvieron resultados válidos.")
            return pd.DataFrame()

        df_final = pd.DataFrame(resultados_globales)
        df_final = df_final.sort_values(by=['P_val_Indirecto'])
        print(f"\n Screening completado con éxito.\n")
        return df_final

# MODELOS:  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
# https://pingouin-stats.org/generated/pingouin.mediation_analysis.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
