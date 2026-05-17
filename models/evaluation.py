"""
MÓDULO DE EVALUACIÓN Y AUDITORÍA DE MODELOS
-------------------------------------------
Este script genera las tablas maestras finales para el TFM:
1. Selecciona el mejor modelo por dominio/modalidad (auditando sobreajuste).
2. Extrae la importancia de las variables.
3. Identifica los Top 5 Biomarcadores cerebrales (residuos puros).
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
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
        self.ruta_mediacion = self.graphics_dir / "GRAFICAS_MEDIACION"
        self.brain_images_dir = self.graphics_dir / "BRAIN_IMAGES"
        self.cortical_dir = self.brain_images_dir / "CORTICAL_MORFOMETRÍA"
        self.subcortical_dir = self.brain_images_dir / "SUBCORTICAL_VOL_RAD"
        
        self.benchmarker = ModelBenchmarker(proyecto_dir)

        self.cortical_dir.mkdir(parents=True, exist_ok=True)
        self.subcortical_dir.mkdir(parents=True, exist_ok=True)
        self.ruta_mediacion.mkdir(parents=True, exist_ok=True)

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

def generar_cortical_brain_tfm_final(self, df_maestra, dominio, modalidad):
        """Genera el volumen NIfTI mapeando las regiones corticales del atlas Harvard-Oxford."""
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = image.load_img(atlas.maps)
        atlas_data = atlas_img.get_fdata().astype(int)
        labels = atlas.labels

        traductor_final = {
            'caudalmiddlefrontal': 'Middle Frontal Gyrus', 'rostralmiddlefrontal': 'Middle Frontal Gyrus',
            'superiorfrontal': 'Superior Frontal Gyrus', 'parsopercularis': 'Inferior Frontal Gyrus, pars opercularis',
            'parsorbitalis': 'Frontal Orbital Cortex', 'medialorbitofrontal': 'Frontal Orbital Cortex',
            'paracentral': 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
            'isthmuscingulate': 'Cingulate Gyrus, posterior division', 'posteriorcingulate': 'Cingulate Gyrus, posterior division',
            'caudalanteriorcingulate': 'Cingulate Gyrus, anterior division', 'rostralanteriorcingulate': 'Cingulate Gyrus, anterior division',
            'insula': 'Insular Cortex', 'transversetemporal': "Heschl's Gyrus (includes H1 and H2)",
            'inferiorparietal': 'Supramarginal Gyrus, anterior division', 'fusiform': 'Temporal Occipital Fusiform Cortex',
            'lingual': 'Lingual Gyrus', 'lateraloccipital': 'Lateral Occipital Cortex, superior division',
            'parahippocampal': 'Parahippocampal Gyrus, anterior division'
        }

        fila = df_maestra[(df_maestra['Dominio'] == dominio) & (df_maestra['Modalidad'] == modalidad)].iloc[0]
        raw_biomarkers = fila['Top 5 Biomarcadores (%)'].split(' | ')
        
        volumen_calor = np.zeros(atlas_data.shape)
        centro_x = atlas_data.shape[0] // 2 

        for b in raw_biomarkers:
            name_fs = b.split(' (')[0].strip()
            valor = float(b.split('(')[1].replace('%)', ''))
            
            es_izquierdo = name_fs.startswith('lh_')
            es_derecho = name_fs.startswith('rh_')
            
            region_clean = name_fs.replace('lh_', '').replace('rh_', '').split('_')[0].lower()
            nombre_atlas = traductor_final.get(region_clean)

            if nombre_atlas in labels:
                idx = labels.index(nombre_atlas)
                mask_region = (atlas_data == idx)
                
                if es_izquierdo:
                    volumen_calor[centro_x:, :, :][mask_region[centro_x:, :, :]] = valor
                elif es_derecho:
                    volumen_calor[:centro_x, :, :][mask_region[:centro_x, :, :]] = valor
                else:
                    volumen_calor[mask_region] = valor

        nifti_final = nib.Nifti1Image(volumen_calor, atlas_img.affine)
        
        # Guardar corte axial base
        plt.figure(figsize=(10, 4))
        plotting.plot_stat_map(nifti_final, display_mode='z', cut_coords=7, 
                               title=f'Cortes Corticales: {dominio}', cmap='hot', threshold=0.1, output_file=None)
        path_img = self.cortical_dir / f"Axial_{dominio.upper()}.png"
        plt.savefig(path_img, dpi=300, bbox_inches='tight')
        plt.close()
        
        return nifti_final

    def generar_vista_superficie_3d(self, nifti_final, dominio):
        """Proyecta el mapa volumétrico a una malla 3D fsaverage y guarda la imagen."""
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
        texture_left = surface.vol_to_surf(nifti_final, fsaverage.pial_left)
        texture_right = surface.vol_to_surf(nifti_final, fsaverage.pial_right)
        
        fig = plt.figure(figsize=(16, 8))
        
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi='left', title=f'H. Izquierdo - {dominio}',
                                    colorbar=True, cmap='hot', threshold=0.1, bg_map=fsaverage.sulc_left, axes=ax1)
        
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi='right', title=f'H. Derecho - {dominio}',
                                    colorbar=True, cmap='hot', threshold=0.1, bg_map=fsaverage.sulc_right, axes=ax2)
        
        path_save = self.cortical_dir / f"3D_Cortical_{dominio.upper()}.png"
        plt.savefig(path_save, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" [Morfometría Cortical] Guardado: {path_save.name}")

    def generar_mapa_total_cerebro(self, df_maestra, dominio, modalidad):
        """Mapea estructuras profundas (Ganglios Basales) mediante el atlas subcortical."""
        atlas_s = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
        atlas_img = image.load_img(atlas_s.maps)
        data_s = atlas_img.get_fdata().astype(int)
        labels_s = atlas_s.labels

        traductor_subcortical = {
            'putamen': 'Putamen', 'palido': 'Pallidum', 'pallidum': 'Pallidum',
            'caudate': 'Caudate', 'accumbens': 'Accumbens', 'amygdala': 'Amygdala',
            'thalamus': 'Thalamus', 'hippocampus': 'Hippocampus', 'ventriculos_lat': 'Lateral Ventricle',
            'lateral ventricle': 'Lateral Ventricle', 'cerebral cortex': 'Cerebral Cortex',
            'cerebral white matter': 'Cerebral White Matter', 'brain-stem': 'Brain-Stem'
        }

        fila = df_maestra[(df_maestra['Dominio'] == dominio) & (df_maestra['Modalidad'] == modalidad)].iloc[-1]
        raw_biomarkers = [b for b in fila['Top 5 Biomarcadores (%)'].split(' | ') if float(b.split('(')[1].replace('%)', '')) > 0]
            
        volumen_calor = np.zeros(data_s.shape)

        for b in raw_biomarkers:
            name_fs = b.split(' (')[0].strip().lower()
            valor = float(b.split('(')[1].replace('%)', ''))

            region_clean = name_fs.replace('left ', '').replace('right ', '').replace('area', '').strip().split('_')[0]
            nombre_base = traductor_subcortical.get(region_clean)
            
            if nombre_base:
                hemi_pref = "Left" if ('left' in name_fs or '_l' in name_fs) else "Right"
                nombre_full = f"{hemi_pref} {nombre_base}"
                
                if nombre_full in labels_s:
                    idx = labels_s.index(nombre_full)
                    volumen_calor[data_s == idx] = valor

        nifti_final = nib.Nifti1Image(volumen_calor, atlas_img.affine)

        # Lienzo unificado estructural
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
        
        plotting.plot_glass_brain(nifti_final, display_mode='lyrz', colorbar=True, cmap='hot', 
                                  threshold=0.1, axes=ax1, title=f'Cerebro de Cristal: {dominio} ({modalidad})')

        plotting.plot_stat_map(nifti_final, display_mode='z', cut_coords=[-10, 0, 10, 20], colorbar=True, 
                               cmap='hot', threshold=0.1, axes=ax2, title='Localización en Ganglios Basales (Cortes Z)')
        
        path_save = self.subcortical_dir / f"Glass_y_Z_{dominio.upper()}_{modalidad.upper()}.png"
        plt.savefig(path_save, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return nifti_final

    def visualizar_biomarcadores_top_calidad(self, nifti_final, dominio, modalidad):
        """Genera y guarda vistas ortogonales cruzadas de alta resolución anatómica."""

        plt.figure(figsize=(15, 10))
        display = plotting.plot_stat_map(
            nifti_final, display_mode='ortho', draw_cross=True, colorbar=True,
            title=f"Localización Anatómica Precisa: {dominio} ({modalidad})",
            cmap='black_red', threshold=0.1, annotate=True
        )
        display.add_contours(nifti_final, levels=[0.5], colors='yellow', linewidths=0.5)
        
        path_ortho = self.subcortical_dir / f"Ortho_{dominio.upper()}_{modalidad.upper()}.png"
        plt.savefig(path_ortho, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Volumen y Radiómica Subcortical] Guardado: {path_ortho.name}")

    def ejecutar_renderizado_lote_neuroimagen(self, df_maestra_clf, df_maestra_reg):
        """Orquesta de forma masiva la exportación de mapas anatómicos para todos los dominios."""
        df_completa = pd.concat([df_maestra_clf, df_maestra_reg], ignore_index=True)
        dominios = df_completa['Dominio'].unique()
        
        print(f"\nIniciando generación masiva de mapas neuroanatómicos para {len(dominios)} dominios...")

        for dominio in dominios:
            # Ejecución Cortical (Morfometría)
            existe_morf = df_completa[(df_completa['Dominio'] == dominio) & (df_completa['Modalidad'] == 'Morfometria')]
            if not existe_morf.empty:
                try:
                    nifti_cort = self.generar_cortical_brain_tfm_final(df_completa, dominio, 'Morfometria')
                    self.generar_vista_superficie_3d(nifti_cort, dominio)
                except Exception as e:
                    print(f" X Error en mapas corticales de {dominio}: {e}")

            # Ejecución Subcortical (Volumen y Radiómica)
            for mod in ['Volumen', 'Radiómica']:
                existe_sub = df_completa[(df_completa['Dominio'] == dominio) & (df_completa['Modalidad'] == mod)]
                if not existe_sub.empty:
                    try:
                        nifti_sub = self.generar_mapa_total_cerebro(df_completa, dominio, mod)
                        if nifti_sub is not None:
                            self.visualizar_biomarcadores_top_calidad(nifti_sub, dominio, mod)
                    except Exception as e:
                        print(f"X Error en mapas subcorticales de {dominio} ({mod}): {e}")

        print("\n ¡Todos los mapas e imágenes de neuroimagen han sido exportados con éxito!")

def generar_reporte_top_mediadores_v7(self, df_resultados, dict_datasets, x_var='crisis_encefalop_ticas_v2', p_umbral=0.15):
        """
        Filtra los mediadores significativos, genera las gráficas de Ruta A y Ruta B,
        y las guarda automáticamente en la carpeta del proyecto del TFM de forma limpia.
        """
        col_path_a = [c for c in df_resultados.columns if 'P_Path_A' in c][0]
        col_path_b = [c for c in df_resultados.columns if 'P_Path_B' in c][0]

        # Filtramos por el p-valor deseado (< 0.15)
        df_filtrado = df_resultados[df_resultados['P_val_Indirecto'] < p_umbral].sort_values('P_val_Indirecto')
        
        colores_mod = {
            'volumen': 'Blues', 'morfometria': 'Greens', 'radiómica': 'Oranges', 'radiomica': 'Oranges'}

        print(f" Guardando gráficos vectoriales en: {self.ruta_mediacion}...\n")

        for idx, row in df_filtrado.iterrows():
            mod = row['Modalidad_M']
            roi = row['Mediador_M']
            target = row['Target_Y']
            
            if mod not in dict_datasets:
                print(f"La modalidad '{mod_raw}' no se encuentra en datasets_listos. Llaves disponibles: {list(dict_datasets.keys())}")
                continue
            
            df_mod = dict_datasets[mod].copy()
            df_clin = dict_datasets["Clínico"].copy()
            
            try:
                df_master = pd.merge(df_mod[['record_id', roi]], 
                                     df_clin[['record_id', x_var, target, 'sexo', 'Edad_RM']], 
                                     on='record_id').dropna()
                
                # Residuos
                df_res = self.benchmarker.ajustar_por_residuos(df_master, [roi])
                
                # Figura
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f"MODALIDAD: {mod_raw.upper()} | {roi} -> {target}", 
                             fontsize=13, fontweight='bold', color='darkgreen' if 'morf' in mod else 'darkblue')

                palette_name = colores_mod.get(mod, 'Set2')

                # ---> Boxplot de Variable Clínica vs Parámetro Cerebral Ajustado
                sns.boxplot(ax=axes[0], data=df_res, x=x_var, y=roi, hue=x_var, palette=palette_name, showfliers=False, legend=False)
                sns.stripplot(ax=axes[0], data=df_res, x=x_var, y=roi, color="black", alpha=0.4)
                axes[0].set_title(f"Ruta A: {x_var} -> Cerebro (p={row[col_path_a]:.3f})", fontsize=11)
                axes[0].set_ylabel(f"{roi} (Residuos ajustados por Edad/Sexo)")

                # ---> Regresión del Parámetro Cerebral Ajustado vs Dominio Neurocognitivo ---
                sns.regplot(ax=axes[1], data=df_res, x=roi, y=target, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                axes[1].set_title(f"Ruta B: Cerebro -> {target} (p={row[col_path_b]:.3f})", fontsize=11)
                axes[1].set_xlabel(f"{roi} (Residuos ajustados)")
                axes[1].set_ylabel(f"Puntuación Z ({target})")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # --- CONTROL DE NOMENCLATURA Y PERSISTENCIA AUTOMÁTICA ---
                fn_sanitizado = f"MEDIACION_{x_var.upper()}_{mod_raw.upper()}_{roi}_{target}".replace(" ", "_").replace("(", "").replace(")", "")
                ruta_archivo_final = self.ruta_mediacion / f"{fn_sanitizado}.png"
                
                # Guardar en alta resolución para la memoria del TFM
                plt.savefig(ruta_archivo_final, dpi=300, bbox_inches='tight')
                plt.close(fig)  # Cierra la figura para liberar memoria RAM
                print(f" Guardado con éxito: {ruta_archivo_final.name}")
                
            except Exception as e:
                print(f"X Error procesando el mediador {roi}: {e}")
        
        print(f" Reportes finales guardados en {self.graphics_dir}")
        return df_clf_fin, df_reg_fin
