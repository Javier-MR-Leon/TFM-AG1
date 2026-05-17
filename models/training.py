"""
MÓDULO DE MACHINE LEARNING - TRAINING & BENCHMARKING
----------------------------------------------------
Este script realiza el entrenamiento masivo buscando el mejor modelo 
para cada dominio neurocognitivo.
"""

import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import pingouin as pg

# Scikit-learn Base
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression

# Modelos (Clasificación y Regresión)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class ModelBenchmarker:
    def __init__(self, proyecto_dir):
        self.proyecto_dir = Path(proyecto_dir)
        self.features_dir = self.proyecto_dir / "3_FEATURES"
        self.graphics_dir = self.proyecto_dir / "4_GRAPHICS"
        self.clean_dir = self.features_dir / "DATA_CLEAN"

    def cargar_datasets_maestros(self):
        """Busca y carga los archivos dataset_ML_*_FINAL.csv"""
        datasets = {}
        archivos = list(self.features_dir.glob("dataset_ML_*_FINAL.csv"))
        
        if not archivos:
            print(f"⚠️ No se encontraron archivos finales en {self.features_dir}")
            return datasets

        for arc in archivos:
            nombre = arc.stem.replace("dataset_ML_", "").replace("_FINAL", "")
            df = pd.read_csv(arc)
            # Normalización de ID y tipos
            if 'id' in df.columns: df['id'] = df['id'].astype(str)
            cols_num = df.select_dtypes(include=[np.number]).columns
            df[cols_num] = df[cols_num].astype(float)
            datasets[nombre] = df
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
        res_clf, res_reg = [], []
        loo = LeaveOneOut()
        conf_clf, conf_reg = self.obtener_configs()

        for nombre_mod, df in dict_datasets.items():
            targets = [c for c in df.columns if c.startswith('D_')]
            cols_cerebrales = df.select_dtypes(include=[np.number]).drop(
                columns=targets + ['id', 'Edad_RM', 'sexo'], errors='ignore'
            ).columns.tolist()

            print(f" Ajustando {nombre_mod} por residuos...")
            df_limpio = self.ajustar_por_residuos(df, cols_cerebrales)

            for target in targets:
                df_estudio = df_limpio.dropna(subset=[target]).copy()
                if len(df_estudio) < 15: continue

                X = df_estudio[cols_cerebrales]
                y_clf = (df_estudio[target] < -1.5).astype(int)
                y_reg = df_estudio[target]

                print(f" ---> Optimizando CLF y REG para {target} en {nombre_mod}...")

                # CLASIFICACIÓN
                for conf in conf_clf:
                    pipe = Pipeline([('scaler', StandardScaler()), ('selector', SelectKBest(f_classif)), ('clf', conf['m'])])
                    grid = GridSearchCV(pipe, conf['p'], cv=loo, scoring='accuracy', n_jobs=-1, return_train_score=True)
                    grid.fit(X, y_clf)
                    res_clf.append({
                        'Modalidad': nombre_mod, 'Dominio': target, 'Modelo': conf['n'],
                        'Acc_Val (%)': round(grid.best_score_ * 100, 2),
                        'Gap_Sobreajuste (%)': round((grid.cv_results_['mean_train_score'][grid.best_index_] - grid.best_score_) * 100, 2),
                        'Mejor_K': grid.best_params_.get('selector__k')
                    })

                # REGRESIÓN
                for conf in conf_reg:
                    pipe = Pipeline([('scaler', StandardScaler()), ('selector', SelectKBest(f_regression)), ('clf', conf['m'])])
                    grid = GridSearchCV(pipe, conf['p'], cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1, return_train_score=True)
                    grid.fit(X, y_reg)
                    res_reg.append({
                        'Modalidad': nombre_mod, 'Dominio': target, 'Modelo': conf['n'],
                        'MAE_Val (Z)': round(abs(grid.best_score_), 3),
                        'Gap_Sobreajuste (Z)': round(abs(grid.cv_results_['mean_train_score'][grid.best_index_]) - abs(grid.best_score_), 3),
                        'Mejor_K': grid.best_params_.get('selector__k')
                    })

        # Guardado de tablas en 4_GRAPHICS
        pd.DataFrame(res_clf).to_csv(self.features_dir / "benchmarking_clasificacion.csv", index=False)
        pd.DataFrame(res_reg).to_csv(self.features_dir / "benchmarking_regresion.csv", index=False)
        return pd.DataFrame(res_clf), pd.DataFrame(res_reg)

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
            tabla_mae = df_m.pivot(index='Dominio', columns='Modalidad', values='MAE_Val (Z)')
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(tabla_mae, annot=True, cmap="RdYlGn_r", fmt=".3f", cbar_kws={'label': 'MAE (Error en Z)'})
            plt.title(f"Benchmarking Regresión: {m}\n(Error MAE en Puntuación Z)", fontsize=14, pad=20)
            
            nombre_arch = f"RESULTADOS_REG_{m.replace(' ', '_')}.png"
            plt.savefig(self.graphics_dir / nombre_arch, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\n IMÁGENES GUARDADAS en: {self.graphics_dir}")

def screening_mediacion_crisis_total_tfm(self, dict_datasets, x_var='crisis_encefalop_ticas_v2'):
        """X: Variable clínica de Crisis | M: Mediadores de Imagen | Y: Dominios Cognitivos"""
        print(f"\n INICIANDO MEGA-SCREENING MULTITAREA (X = {x_var})")

        df_clin = dict_datasets["Clínico"].copy()
        df_clin['record_id'] = df_clin['record_id'].astype(str).str.strip()
        targets_y = [c for c in df_clin.columns if c.startswith('D_')]

        resultados_globales = []
        datasets_analizar = ["Volumen", "Morfometria", "Radiómica"]
        
        for tipo_ds in datasets_analizar:
            
            print(f"Procesando Dominio Imagen: {tipo_ds}...")
            df_brain = dict_datasets[tipo_ds].copy()
            df_brain['record_id'] = df_brain['record_id'].astype(str).str.strip()
            
            cols_drop = [c for c in df_brain.columns if c.startswith('D_') or c in ['sexo', 'Edad_RM']]
            df_brain_clean = df_brain.drop(columns=[c for c in cols_drop if c != 'record_id'])
            df_master = pd.merge(df_brain_clean, df_clin, on='record_id')
            
            mediadores = [c for c in df_brain_clean.columns if c not in ['record_id', 'sexo', 'Edad_RM', x_var, 'crisis_encefalop_ticas_v2', 'bioquimico']]

            for target in targets_y:
                for m_roi in mediadores:
                    try:
                        df_model = df_master[['crisis_encefalop_ticas_v2', m_roi, target, 'sexo', 'Edad_RM']].dropna()
                        if len(df_model) < 15: continue
                        
                        # Corrección: Uso del método de residuos interno de la clase
                        df_res = self.ajustar_por_residuos(df_model, [m_roi])
                        res = pg.mediation_analysis(data=df_res, x='crisis_encefalop_ticas_v2', m=m_roi, y=target, n_boot=200, seed=42)
                        
                        col_c = [c for c in res.columns if 'coef' in c.lower()][0]
                        col_p = [c for c in res.columns if 'pval' in c.lower() or 'p-val' in c.lower()][0]
                        
                        row_a = res[res['path'].str.contains('~ X', na=False)]
                        row_b = res[res['path'].str.contains('Y ~', na=False)]
                        row_ind = res[res['path'].str.contains('Indirect', na=False)]
                        
                        resultados_globales.append({
                            'Target_Y': target, 'Modalidad_M': tipo_ds, 'Mediador_M': m_roi,
                            'Coef_Indirecto': row_ind[col_c].values[0], 'P_val_Indirecto': row_ind[col_p].values[0],
                            'Sig': row_ind['sig'].values[0],
                            'P_Path_A (Crisis->Medida)': row_a[col_p].values[0],
                            'P_Path_B (Medida->Dominio)': row_b[col_p].values[0]
                        })
                    except: continue

        df_final = pd.DataFrame(resultados_globales)
        df_final = df_final.sort_values(by=['P_val_Indirecto'])
        print(f" \nScreening Crisis completado.\n")
        return df_final

    def screening_mediacion_bioq_total_tfm(self, dict_datasets, x_var='bioquimico'):
        """X: Variable clínica Bioquímica | M: Mediadores de Imagen | Y: Dominios Cognitivos"""
        print(f"\nINICIANDO MEGA-SCREENING MULTITAREA (X = {x_var})")

        df_clin = dict_datasets["Clínico"].copy()
        df_clin['record_id'] = df_clin['record_id'].astype(str).str.strip()
        targets_y = [c for c in df_clin.columns if c.startswith('D_')]

        resultados_globales = []
        datasets_analizar = ["Volumen", "Morfometria", "Radiómica"]
        
        for tipo_ds in datasets_analizar:
            if tipo_ds not in dict_datasets: continue
            
            print(f"Procesando Dominio Imagen: {tipo_ds}...")
            df_brain = dict_datasets[tipo_ds].copy()
            df_brain['record_id'] = df_brain['record_id'].astype(str).str.strip()
            
            cols_drop = [c for c in df_brain.columns if c.startswith('D_') or c in ['sexo', 'Edad_RM']]
            df_brain_clean = df_brain.drop(columns=[c for c in cols_drop if c != 'record_id'])
            df_master = pd.merge(df_brain_clean, df_clin, on='record_id')
            
            mediadores = [c for c in df_brain_clean.columns if c not in ['record_id', 'sexo', 'Edad_RM', x_var, 'bioquimico', 'crisis_encefalop_ticas_v2']]

            for target in targets_y:
                for m_roi in mediadores:
                    try:
                        df_model = df_master[['bioquimico', m_roi, target, 'sexo', 'Edad_RM']].dropna()
                        if len(df_model) < 15: continue
                        
                        # Corrección: Uso del método de residuos interno de la clase
                        df_res = self.ajustar_por_residuos(df_model, [m_roi])
                        res = pg.mediation_analysis(data=df_res, x='bioquimico', m=m_roi, y=target, n_boot=200, seed=42)
                        
                        col_c = [c for c in res.columns if 'coef' in c.lower()][0]
                        col_p = [c for c in res.columns if 'pval' in c.lower() or 'p-val' in c.lower()][0]
                        
                        row_a = res[res['path'].str.contains('~ X', na=False)]
                        row_b = res[res['path'].str.contains('Y ~', na=False)]
                        row_ind = res[res['path'].str.contains('Indirect', na=False)]
                        
                        resultados_globales.append({
                            'Target_Y': target, 'Modalidad_M': tipo_ds, 'Mediador_M': m_roi,
                            'Coef_Indirecto': row_ind[col_c].values[0], 'P_val_Indirecto': row_ind[col_p].values[0],
                            'Sig': row_ind['sig'].values[0],
                            'P_Path_A (Bioquímico->Medida)': row_a[col_p].values[0],
                            'P_Path_B (Medida->Dominio)': row_b[col_p].values[0]
                        })
                    except: continue

        df_final = pd.DataFrame(resultados_globales)
        df_final = df_final.sort_values(by=['P_val_Indirecto'])
        print(f"\nScreening Bioquímico completado.\n")
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
