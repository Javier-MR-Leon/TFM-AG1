# TFM-AG1: Pipeline Multimodal de Neuroimagen y Machine Learning

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Docker](https://img.shields.io/badge/docker-required-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Este repositorio contiene el pipeline completo de procesamiento y análisis de datos desarrollado para el **Trabajo de Fin de Máster (TFM) en Ciencia de Datos**. El proyecto integra técnicas de Deep Learning para el procesamiento de imágenes de Resonancia Magnética (RM), preprocesamiento de datos y Machine Learning para la predicción de dimensiones neurocognitivas en pacientes con Aciduria Glutárica tipo 1 (AG1) mediante modelos de clasificación y regresión.

----

## Descripción del Proyecto

El pipeline transforma imágenes brutas **DICOM** en datos normalizados que serán utilizados para desarrollar conocimiento clínico cuantitativo. El pipeline extraerá datos por tres bloques estucturales (Volumetría, Grosor y Radiómica (más Clínica)) y los usará en modelos de machine learning supervisados con auditoría estricta de **sobreajuste (Overfitting)** y mediante validación cruzada **Leave-One-Out (LOOCV)**.

### Capacidades Clave
* **Triaje Técnico:** Filtro automático de imágenes basado en resolución de matriz y calidad anatómica.
* **Procesamiento Dockerizado:** Integración de `SynthSeg` y `FastSurfer` para segmentaciones subcorticales y corticales robustas.
* **Análisis por Residuos:** Ajuste multivariable para eliminar la varianza explicada por la **Edad** y el **Sexo**, aislando el efecto biológico puro.
* **Benchmarking Masivo:** Entrenamiento simultáneo de +10 modelos de Clasificación y Regresión para identificar la arquitectura óptima por cada dominio cognitivo.

---

## Estructura del Repositorio

```text
TFM-AG1/
├── data/               # (Local) Datos DICOM, Excel clínico y resultados.
├── src/                # Fase 1: Ingeniería de Características
│   ├── preprocessing_img.py    # DICOM a NIfTI y organización T1.
│   ├── structure.py            # Construcción del árbol de directorios.
│   ├── segmentation.py         # Orquestación de contenedores Docker.
│   ├── preprocessing_feat.py   # Normalización, EDA y Poda de colinealidad.
│   └── main.py                 # Orquestador maestro de imagen.
├── models/             # Fase 2: Machine Learning
│   ├── training.py             # Benchmarking LOOCV y ajuste por residuos.
│   ├── evaluation.py           # Auditoría de modelos y extracción de Top 5.
│   └── main.py                 # Orquestador maestro de ML.
├── requirements.txt    # Dependencias del entorno.
└── .gitignore          # Exclusión de archivos .nii.gz, .csv y .xlsx.
```

## Requisitos e Instalación

### 1. Entorno de Python
Se recomienda el uso de un entorno virtual (**Conda**) para garantizar la compatibilidad de las librerías.
* **Versión recomendada:** Python 3.10.19
* **Instalación de dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Software Externo (Crucial)
Este pipeline utiliza la tecnología de contenedores para asegurar que el procesamiento de imágenes sea idéntico en cualquier máquina:

* **Docker Desktop:** Debe estar instalado, configurado y en ejecución antes de lanzar el pipeline de imagen. Es el motor que permite ejecutar `SynthSeg` y `FastSurfer` sin instalar dependencias complejas.
    * [Descargar Docker Desktop aquí](https://www.docker.com/products/docker-desktop/)
* **Licencia de FreeSurfer:** Es obligatoria para los procesos de `SynthSeg` y `FastSurfer`. 
    * Consigue tu archivo `license.txt` gratuitamente [aquí](https://surfer.nmr.mgh.harvard.edu/fswiki/Registration).
    * Ubícalo en la carpeta: `./data/LICENSE_FS/license.txt`.

---

## Flujo de Trabajo (Workflow)

El proyecto se divide en dos grandes etapas independientes que se ejecutan secuencialmente:

### Fase 1: Ingeniería de Características de Imagen
Procesa la señal neuroanatómica bruta para convertirla en datos estructurados.
```bash
python src/main.py
```

¿Qué sucede? Se convierten las imágenes, se filtran por calidad, se lanzan las segmentaciones en Docker y se generan los archivos CSV finales normalizados en data/ESTUDIO_TFM/3_FEATURES.

### Fase 2: Machine Learning y Evaluación
Ejecuta el script maestro en models/:
```bash
python models/main.py
```

¿Qué sucede? Se cargan los datos, se elimina el efecto de la Edad/Sexo mediante regresión lineal (residuos), se entrenan múltiples modelos y se generan los reportes de rendimiento y biomarcadores en data/ESTUDIO_TFM/4_GRAPHICS.

## Resultados e Interpretación

El pipeline no solo procesa datos, sino que genera automáticamente los materiales visuales necesarios para la memoria del TFM, localizados en `data/ESTUDIO_TFM/4_GRAPHICS`:

### Salidas Generadas
* **Mapas de Calor (Heatmaps):** Comparativas visuales de rendimiento. Permiten identificar de un vistazo qué combinaciones de *Modelo* (ej. SVM, Lasso) y *Modalidad* (ej. Radiómica, Volumetría) ofrecen mayor **Accuracy** en clasificación y menor **MAE** (Error Absoluto Medio) en regresión.
* **Top 5 Biomarcadores:** Por cada modelo óptimo, el sistema extrae las 5 regiones cerebrales con mayor peso diagnóstico. Los resultados se presentan en porcentaje de importancia relativa, facilitando la interpretación clínica.
* **Informes de Auditoría de Sobreajuste:** Gráficas que comparan el error en entrenamiento vs. validación. El sistema resalta visualmente los modelos con un "Gap" elevado para evitar conclusiones basadas en **Overfitting**.

---

## Autor

| Investigador | Proyecto | Institución |
| :--- | :--- | :--- |
| **Javier MArtínez Rodríguez** | Trabajo de Fin de Máster (TFM) | [UOC] |

---
> **Aviso:** Este repositorio ha sido creado con fines académicos. El código es abierto bajo licencia MIT, pero los datos de pacientes están protegidos y no se incluyen en el historial de versiones.
