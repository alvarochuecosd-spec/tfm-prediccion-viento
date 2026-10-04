# Versionado de datos con DVC

DVC (*Data Version Control*) versiona los ficheros de datos igual que Git versiona el
código: guarda un puntero ligero en el repositorio Git y el contenido real en un
almacenamiento aparte (local, S3, Google Drive...). Cubre el objetivo 5 del Anexo I en
su parte de versionado de datos.

No es código Python — se opera desde la terminal y con ficheros de configuración
(`.dvc`, `dvc.yaml`). Esta guía documenta los comandos exactos para este proyecto.

## Por qué hace falta

Sin DVC, `dataset_modelado.csv` y `harmonie_crudo.csv` (varios GB entre los dos) o bien no
se versionan —y entonces nadie puede reproducir exactamente los resultados del TFM sin
volver a ejecutar todo el ETL— o se suben a Git, que no está pensado para ficheros grandes
y deja el repositorio inmanejable.

## 1. Instalación e inicialización

```bash
pip install dvc

# Dentro del repositorio Git del TFM (asumiendo que ya existe)
dvc init
git add .dvc .dvcignore
git commit -m "Inicializar DVC"
```

## 2. Configurar el almacenamiento remoto

Para un TFM, lo más simple es una carpeta local fuera del repositorio (un disco externo,
una carpeta de Drive sincronizada) o, si se dispone de ello, un bucket S3/GCS.

```bash
# Opción local (más simple para el TFM)
dvc remote add -d almacen /ruta/fuera/del/repo/dvc-storage

# Opción S3 (si se dispone de cuenta AWS)
dvc remote add -d almacen s3://mi-bucket/tfm-viento

git add .dvc/config
git commit -m "Configurar remoto DVC"
```

## 3. Versionar los datasets

```bash
# Los grandes: el crudo de HARMONIE y el dataset ya preparado
dvc add harmonie_crudo.csv
dvc add dataset_modelado.csv
dvc add observaciones_horarias.csv

git add harmonie_crudo.csv.dvc dataset_modelado.csv.dvc observaciones_horarias.csv.dvc .gitignore
git commit -m "Versionar datasets con DVC"

dvc push   # sube el contenido real al remoto configurado
```

A partir de aquí, Git lleva el puntero (`.dvc`, unos pocos KB) y DVC lleva el contenido.

## 4. Pipeline reproducible: `dvc.yaml`

Define el ETL y la preparación como un pipeline con dependencias explícitas, para que
`dvc repro` regenere solo lo que haya cambiado.

```yaml
# dvc.yaml
stages:
  extraccion_harmonie:
    cmd: python extraer_harmonie.py
    deps:
      - extraer_harmonie.py
    outs:
      - harmonie_crudo.csv

  etl:
    cmd: jupyter nbconvert --to notebook --execute 1_ETL.ipynb --output 1_ETL.ipynb
    deps:
      - 1_ETL.ipynb
      - harmonie_crudo.csv
      - Tarifa_Tabla1.dat
      - Tarifa_Tabla1_2022.dat
    outs:
      - dataset_horario.csv
      - observaciones_horarias.csv

  preparacion:
    cmd: jupyter nbconvert --to notebook --execute 2_preparacion_EDA.ipynb --output 2_preparacion_EDA.ipynb
    deps:
      - 2_preparacion_EDA.ipynb
      - dataset_horario.csv
    outs:
      - dataset_modelado.csv
```

```bash
dvc repro          # ejecuta solo las etapas cuyas dependencias cambiaron
dvc dag             # visualiza el grafo de dependencias
```

## 5. Flujo de trabajo diario

```bash
# Tras modificar el ETL y regenerar datos
dvc add dataset_modelado.csv
git add dataset_modelado.csv.dvc
git commit -m "Actualizar dataset: corrección del sesgo direccional"
dvc push

# Al clonar el repositorio en otra máquina (o volver a una versión antigua)
git clone <repo>
git checkout <commit-o-tag>
dvc pull            # descarga la versión exacta de los datos correspondiente a ese commit
```

## 6. Relación con MLflow (notebook 7)

MLflow versiona **modelos y experimentos**; DVC versiona **datos**. Se complementan: un
run de MLflow puede registrar el hash de commit de Git (`mlflow.log_param('git_commit',
...)`) junto con la métrica, de modo que cualquier resultado queda trazable hasta el
dataset exacto (vía DVC) y el código exacto (vía Git) que lo produjeron.

```python
import subprocess
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
mlflow.log_param('git_commit', commit)
mlflow.log_param('dvc_data_hash', open('dataset_modelado.csv.dvc').read())
```

## Para la memoria

Vale la pena incluir en el TFM el grafo de `dvc dag` y una captura del historial de
`dvc.yaml` — es evidencia visual de reproducibilidad que un tribunal valora, y cuesta
minutos generarla una vez el pipeline está en marcha.
