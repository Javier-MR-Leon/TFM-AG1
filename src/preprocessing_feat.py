MÓDULO DE PROCESAMIENTO DE FEATURES Y NORMALIZACIÓN
---------------------------------------------------
Este script realiza:
1. Creación de dominios neurocognitivos (D_).
2. Auditoría de calidad y Sanity Check (salida a 2_QC).
3. Normalización de neuroimagen (ICV para volúmenes, Z-score para el resto). 
4. Poda de variables colineales para reducir dimensionalidad.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class FeatureProcessor:
    def __init__(self, proyecto_dir):
        self.proyecto_dir = proyecto_dir
        self.qc_dir = os.path.join(proyecto_dir, "2_QC")
        self.features_dir = os.path.join(proyecto_dir, "3_FEATURES")
        os.makedirs(self.qc_dir, exist_ok=True)

    def crear_dominios_clinicos(self, ruta_excel):
        """
        Limpia el Excel clínico y genera los 5 Dominios Maestros (D_).
        El excel utilizado es el RAW bajado de REDcap, base de datos del 12 de Octubre. 
        """
        print("\n Generando Dominios Neurocognitivos...")
        df = pd.read_excel(ruta_excel)
        
        # Filtro de inclusión - Han sido marcados con SI, aquellos casos que tienen RM
        df = df[df['para_estudio'].astype(str).str.strip().str.upper() == 'SI'].copy()
        
        # Limpieza
        columnas_a_excluir = ['paciente', 'pacientes', 'para_estudio', 'Anotaciones', 
                             'Edad_naz_registro', 'dps', 'tipo_lesi_n_en_la_rm']
        df = df.drop(columns=[c for c in columnas_a_excluir if c in df.columns]).copy()

        # Convertimos las comas en puntos y nos aseguramos que las columnas clinicas son numéricas.
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                except: pass

        # --- CREACIÓN DE LOS 5 DOMINIOS MAESTROS ---

        df['D_Cognicion_Global'] = df['cd_ci_z_score'].fillna(df['ci_z_score'])
        
        df['D_Lenguaje'] = df['cd_lenguaje_z_score'].fillna(df['ic_z_score'])
      
        media_gp = df[['gp_zscore', 'gp_zscore_2']].mean(axis=1, skipna=True) if 'gp_zscore_2' in df.columns else df['gp_zscore']
        df['D_Motor'] = np.where(df['cd_ci_z_score'].notna(), df['cd_motor_3_zs'], media_gp)
        df['D_Motor'] = df['D_Motor'].fillna(df['cd_motor_3_zs']).fillna(media_gp)

        cols_at = [c for c in ['at_claves_2', 'at_claves_3'] if c in df.columns]
        claves = df[cols_at].mean(axis=1) if cols_at else np.nan
        df['D_Atencion'] = pd.concat([df['TMT_A_Z score'], claves], axis=1).mean(axis=1, skipna=True)

        df['D_Ejecutivo'] = df[['TMT_B_Z score', 'if_z_score']].mean(axis=1, skipna=True)

        # Nos quedamos solo con las columnas que queremos, el resto son eliminadas 
        columnas_finales = ['id', 'Edad_RM', 'bioquimico', 'sexo', 'perimetro_craneal_v2', 
                    'crisis_encefalop_ticas_v2', 'desarrollo_psicomotor_norm_v2', 'RM_L_Displasia_opercular', 
                    'RM_L_Ganglios_Basales', 'RM_L_Sustancia_Blanca', 'RM_Restricciones', 'RM_Nuleos_Dentados_Otros', 
                    'rm_normal2', 'D_Cognicion_Global', 'D_Lenguaje', 'D_Motor', 'D_Atencion', 'D_Ejecutivo']

        columnas_existentes = [c for c in columnas_finales if c in df_reducido.columns]
        df = df[columnas_existentes].copy()

        # 3. Gestión de IDs Duplicados, ya que tenemos varias entradas de RM para un mismo paciente.
        df['id'] = df['id'].astype(int).astype(str)
        sufijo = (df.groupby('id').cumcount() + 1) * 11
        total_veces = df.groupby('id')['id'].transform('count')
        df['id'] = np.where(total_veces > 1, df['id'] + sufijo.astype(str), df['id'])
        df['id'] = pd.to_numeric(df['id'])

        return df

    def ejecutar_sanity_check_maestro(rutas, qc_dir, umbral_z=2.0):
    """
    SANITY CHECK UNIFICADO:
    1. Escanea asimetrías extremas (Volumetría).
    2. Valida rangos biológicos (Grosor).
    3. Detecta redundancias (Radiómica).
    4. Auditoría Multinivel de Outliers (Z-Score).
    """
    print("INICIANDO SANITY CHECK NEUROANATÓMICO MAESTRO \n")
    os.makedirs(qc_dir, exist_ok=True)
    
    # CARGA DE DATOS
    data = {k: pd.read_csv(v) for k, v in rutas.items()}

    # VOLUMETRÍA: ASIMETRÍA Y RELACIÓN SB/ICV
    df_vol = data['vol']
    print(" Analizando Volumetría...")
    
    # Check SB vs ICV
    col_sb, col_icv = 'left cerebral white matter', 'total intracranial'
    if col_sb in df_vol.columns and col_icv in df_vol.columns:
        corr = df_vol[col_sb].corr(df_vol[col_icv])
        print(f"   • Correlación SB vs ICV: {corr:.2f} (Esperado > 0.80)")

    # Escaneo de Asimetrías (>15%)
    columnas_izq = [c for c in df_vol.columns if 'left' in c.lower()]
    for c_izq in columnas_izq:
        c_der = c_izq.lower().replace('left', 'right')
        if c_der in df_vol.columns:
            asim = (df_vol[c_izq] - df_vol[c_der]) / (df_vol[c_izq] + df_vol[c_der])
            outliers = df_vol[asim.abs() > 0.15]['id'].tolist()
            if outliers:
                print(f" ---> Asimetría extrema en {c_izq.replace('left ','')}: {outliers}")

    # GROSOR: RANGO BIOLÓGICO
    df_grosor = data['grosor']
    print("\n Analizando Grosor Cortical...")
    cols_thick = [c for c in df_grosor.columns if 'thick' in c]
    for c in cols_thick:
        fuera = df_grosor[(df_grosor[c] > 5.0) | (df_grosor[c] < 1.0)]['id'].tolist()
        if fuera:
            print(f" X {c} fuera de rango (1-5mm): {fuera}")
            hay_errores_biologicos = True
    
    if not hay_errores_biologicos:
        print(" Todos los grosores dentro del rango biológico (1-5 mm).")
      
    # RADIÓMICA: REDUNDANCIA 
    df_rad = data['radio']
    print("\n Analizando Radiómica...")
    corr_rad = df_rad.select_dtypes(include=[np.number]).corr().abs()
    upper = corr_rad.where(np.triu(np.ones(corr_rad.shape), k=1).astype(bool))
    clones = [c for c in upper.columns if any(upper[c] > 0.98)]
    print(f"   • Variables redundantes (>0.98): {len(clones)}")

    # AUDITORÍA MULTINIVEL (Z-SCORE)
    print("\n Ejecutando Auditoría de Outliers...")
    hallazgos_totales = []
    
    for mod, df in data.items():
        excluir = ['id']
        cols_num = [c for c in df.columns if c not in excluir and df[c].dtype in ['float64', 'int64']]
        
        for c in cols_num:
            m, s = df[c].mean(), df[c].std()
            if s == 0: continue
            z = (df[c] - m) / s
            
            outliers = df[z.abs() >= umbral_z]
            for idx, fila in outliers.iterrows():
                z_val = abs(z[idx])
                hallazgos_totales.append({
                    'Modalidad': mod,
                    'Paciente': fila['id'],
                    'Región': c,
                    'Valor': round(fila[c], 2),
                    'Z-Score': round(z[idx], 2),
                    'Severidad': " CRÍTICO" if z_val >= 3 else " NOTABLE"
                })

    # Guardar informe
    df_informe = pd.DataFrame(hallazgos_totales)
    if not df_informe.empty:
        df_informe.sort_values(by='Z-Score', key=abs, ascending=False, inplace=True)
        df_informe.to_csv(os.path.join(qc_dir, "informe_sanity_check_outliers.csv"), index=False)
        print(f"✅ Informe de {len(df_informe)} desviaciones guardado en 2_QC.")
      
    return data

    def ejecutar_normalizacion_y_limpieza(self, rutas_dict):
        """
        Función Maestra de Normalización:
        1. Volúmenes: Normalización biológica por ICV.
        2. Grosor/Radiómica: Normalización estadística Z-Score.
        3. Clínico: Imputación de Perímetro y Z-Score de variables continuas.
        """
        datasets_procesados = {}

        print("\n INICIANDO NORMALIZACIÓN MULTIMODAL ")

        for modalidad, ruta in rutas_dict.items():
            if not os.path.exists(ruta):
                print(f" Saltando {modalidad}: No se encuentra el archivo.")
                continue

            df = pd.read_csv(ruta)
            
            # VOLUMETRÍA (Normalización por ICV)
            if modalidad == "vol":
                print(f" [{modalidad.upper()}] Aplicando ratio biológico por ICV...")
                col_icv = 'total intracranial'
                if col_icv in df.columns:
                    excluir = ['id', col_icv]
                    cols_a_norm = [c for c in df.columns if c not in excluir and df[c].dtype in ['float64', 'int64']]
                    for col in cols_a_norm:
                        df[col] = df[col] / df[col_icv]
                datasets_procesados[modalidad] = df

            # CLÍNICO (Imputación + Z-Score selectivo)
            elif modalidad == "clinico":
                print(f" [{modalidad.upper()}] Imputando Perímetro y escalando Edad...")
                # Imputación del perímetro craneal (2 datos faltantes)
                col_pc = 'perimetro_craneal_v2'
                if col_pc in df.columns and df[col_pc].isnull().any():
                    df[col_pc] = df[col_pc].fillna(df[col_pc].median())
                
                # Z-Score solo a continuas
                cols_cont = [c for c in ['Edad_RM', col_pc] if c in df.columns]
                scaler = StandardScaler()
                df[cols_cont] = scaler.fit_transform(df[cols_cont])
                datasets_procesados[modalidad] = df

            # GROSOR Y RADIÓMICA (Z-Score completo)
            else:
                print(f"⚖️ [{modalidad.upper()}] Aplicando Z-Score estadístico...")
                excluir = ['id']
                # Escalar todo lo numérico que no sea ID ni Target (D_)
                features = [c for c in df.columns if c not in excluir  
                            and df[c].dtype in ['float64', 'int64']]
                scaler = StandardScaler()
                df[features] = scaler.fit_transform(df[features])
                datasets_procesados[modalidad] = df

            # Guardar versión normalizada
            ruta_salida = ruta.replace(".csv", "_NORM.csv")
            df.to_csv(ruta_salida, index=False)
            print(f"   Guardado: {os.path.basename(ruta_salida)}")

        return datasets_procesados

    def vincular_metadatos_y_targets(self, datasets_dict):
        """
        Une los datasets de imagen normalizados con los metadatos clínicos (Edad, Sexo y Dominios).
        Utiliza los DataFrames cargados en memoria.
        """
        print("\n VINCULANDO METADATOS CLÍNICOS Y TARGETS (D_)")

        # Preparar el DataFrame de metadatos (Edad, Sexo y los Targets D_)
        df_clin = datasets_dict['clinico'].copy()
        
        columnas_interes = ['id', 'Edad_RM', 'sexo', 'D_Cognicion_Global', 
                            'D_Lenguaje', 'D_Motor', 'D_Atencion', 'D_Ejecutivo']
        
        # Filtramos solo las que existan en tu archivo clínico procesado
        cols_finales = [c for c in columnas_interes if c in df_clin.columns]
        df_metadatos = df_clin[cols_finales].copy()
        df_metadatos['id'] = df_metadatos['id'].astype(str)

        datasets_vinc_finales = {}

        # Unir con cada modalidad de imagen
        for mod, df_img in datasets_dict.items():
            if mod == "clinico": continue 
            
            print(f"   Merging metadatos con {mod.upper()}...")
            
            df_img_copy = df_img.copy()
            df_img_copy['id'] = df_img_copy['id'].astype(str)
            
            # Unimos solo pacientes con datos en ambos sitios
            df_unido = pd.merge(df_metadatos, df_img_copy, on='id', how='inner')
            
            # Guardar físicamente para trazabilidad en GitHub
            ruta_save = os.path.join(self.features_dir, f"{mod}_FINAL_CON_TARGETS.csv")
            df_unido.to_csv(ruta_save, index=False)
            
            datasets_vinc_finales[mod] = df_unido
            print(f" Dataset listo: {os.path.basename(ruta_save)} | Sujetos: {df_unido.shape[0]}")

        return datasets_vinc_finales

    def ejecutar_eda_completo(self, datasets_dict):
        """
        Análisis Exploratorio de Datos (EDA) con guardado selectivo:
        - Gráficas (.png) -> ./data/ESTUDIO_TFM/4_GRAPHICS
        - Informes (.csv) -> ./data/ESTUDIO_TFM/2_QC
        """
        print("\n GENERANDO ANÁLISIS EXPLORATORIO Y GRÁFICAS ")
        
        # Crear carpeta de gráficas si no existe
        graphics_dir = os.path.join(self.proyecto_dir, "4_GRAPHICS")
        os.makedirs(graphics_dir, exist_ok=True)

        for nombre, df in datasets_dict.items():
            print(f" Procesando EDA para: {nombre}")
            
            # Identificar Targets y Variables Numéricas
            targets = [c for c in df.columns if c.startswith('D_')]
            df_num = df.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')
            
            # MAPA DE CALOR DE COLINEALIDAD
            corr_matrix = df_num.corr().abs()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, cmap='YlOrRd', cbar_kws={'label': 'Corr. Absoluta'})
            plt.title(f"Matriz de Colinealidad: {nombre}")
            
            # Guardar Gráfica
            plt.savefig(os.path.join(graphics_dir, f"heatmap_colinealidad_{nombre}.png"), bbox_inches='tight')
            plt.close()

            # Guardar Informe en 2_QC
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            colin_report = []
            for col in upper.columns:
                redundantes = upper.index[upper[col] > 0.85].tolist()
                for r in redundantes:
                    colin_report.append({'Variable_A': col, 'Variable_B': r, 'Correlacion': round(upper.loc[r, col], 4)})
            
            if colin_report:
                pd.DataFrame(colin_report).to_csv(os.path.join(self.qc_dir, f"informe_colinealidad_{nombre}.csv"), index=False)

            # DISTRIBUCIÓN DE TARGETS Y BOXPLOTS
            if targets:
                # 1. Histogramas de distribución
                fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4))
                if len(targets) == 1: axes = [axes]
                for i, target in enumerate(targets):
                    sns.histplot(df[target], kde=True, ax=axes[i], color='skyblue')
                    axes[i].axvline(-1.5, color='red', linestyle='--', label='Umbral Déficit')
                    axes[i].set_title(f"Distribución {target}")
                plt.tight_layout()
                plt.savefig(os.path.join(graphics_dir, f"distribucion_targets_{nombre}.png"))
                plt.close()

                # 2. Boxplots: Diferencia de Grupos (Déficit vs Normal)
                main_t = targets[0] 
                df_temp = df.copy()
                df_temp['Categoria'] = np.where(df_temp[main_t] < -1.5, 'Déficit', 'Normal')
                
                # Top 10 variables con más correlación con el target principal
                top_vars = df_num.corr()[main_t].abs().drop(targets, errors='ignore').sort_values(ascending=False).index[:10]

                plt.figure(figsize=(20, 10))
                for i, var in enumerate(top_vars):
                    plt.subplot(2, 5, i+1)
                    sns.boxplot(x='Categoria', y=var, data=df_temp, palette='Set2', hue='Categoria', legend=False)
                    sns.stripplot(x='Categoria', y=var, data=df_temp, color='black', alpha=0.3, jitter=True)
                    plt.title(f"{i+1}: {var}", fontsize=9)
                plt.suptitle(f"Análisis Diferencial: {nombre} vs {main_t}", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(graphics_dir, f"boxplots_comparativos_{nombre}.png"))
                plt.close()

            # MAPA DE DATOS FALTANTES
            plt.figure(figsize=(12, 4))
            sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
            plt.title(f"Mapa de Datos Faltantes: {nombre}")
            plt.savefig(os.path.join(graphics_dir, f"missing_data_map_{nombre}.png"))
            plt.close()

        print(f" EDA finalizado. \n Gráficas en: {graphics_dir} \n Informes en: {self.qc_dir}")
    
    def ejecutar_poda_multimodal(self, datasets_dict):
        """
        Aplica la poda de colinealidad con umbrales específicos:
        - Radiómica: 0.98 (Para mantener la sensibilidad de textura).
        - Resto (Volumen, Grosor, Clínico): 0.85 (Estándar para evitar redundancia).
        """
        print("\n INICIANDO PODA DE VARIABLES COLINEALES (DINÁMICA)")
        datasets_podados = {}

        for nombre, df in datasets_dict.items():
            # Definir umbral según modalidad
            umbral = 0.98 if "radio" in nombre.lower() else 0.85
            print(f" Procesando {nombre.upper()} (Umbral: {umbral})...")

            # Identificar qué NO se debe tocar
            targets = [c for c in df.columns if c.startswith('D_')]
            id = ['id']
            ajuste = ['Edad_RM', 'sexo']
            
            no_tocar = targets + id + ajuste
            features = [c for c in df.columns if c not in no_tocar and df[c].dtype in ['float64', 'int64']]

            # Matriz de correlación
            df_num = df[features]
            corr_matrix = df_num.corr().abs()
            
            # Máscara para la mitad superior de la matriz - PAra identificar los pares a eliminar
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = []
            registro = []
            
            for col in upper.columns:
                # Buscamos si esta columna supera el umbral con alguna ya procesada
                redundantes = upper.index[upper[col] > umbral].tolist()
                if redundantes:
                    to_drop.append(col)
                    registro.append({
                        'Modalidad': nombre,
                        'Variable_Eliminada': col, 
                        'Mantenida_Referencia': redundantes[0], 
                        'Correlacion': round(upper.loc[redundantes[0], col], 4)
                    })
            
            # Crear el DF final quitando las redundantes
            df_final = df.drop(columns=to_drop)
            datasets_podados[nombre] = df_final
            
            # Guardar informe técnico en 2_QC
            if registro:
                ruta_informe = os.path.join(self.qc_dir, f"poda_registro_{nombre}.csv")
                pd.DataFrame(registro).to_csv(ruta_informe, index=False)
            
            # Guardar el dataset final de entrenamiento en 3_FEATURES
            ruta_final = os.path.join(self.features_dir, f"dataset_ML_{nombre}_FINAL.csv")
            df_final.to_csv(ruta_final, index=False)
            
            print(f" {nombre.upper()}: Eliminadas {len(to_drop)} variables redundantes.")
            print(f" Dimensiones finales: {df_final.shape[1]} columnas.")

        return datasets_podados
