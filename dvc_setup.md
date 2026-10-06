# Versionado de datos con DVC

Los datos del trabajo no se publican en el repositorio, ya que se han utilizado con permiso
de sus propietarios y no son de distribución pública. Sí están versionados con DVC: el
repositorio incluye los ficheros de referencia `.dvc`, que guardan el hash MD5 y el tamaño de
cada fichero de datos. Así, cada versión del código queda ligada a una versión concreta de los
datos (Sección 4.7 de la memoria).

| Fichero de datos | Referencia DVC | Tamaño | Lo genera |
|---|---|---|---|
| `harmonie_crudo.csv` | `harmonie_crudo.csv.dvc` | 52 MB | Extracción de los ficheros NetCDF de HARMONIE-AROME |
| `observaciones_horarias.csv` | `observaciones_horarias.csv.dvc` | 4,6 MB | Notebook 1 (agregación horaria del anemómetro) |
| `dataset_modelado.csv` | `dataset_modelado.csv.dvc` | 9,2 MB | Notebook 2 |

El resto de ficheros intermedios (`dataset_horario.csv`, predicciones en `*.parquet`) se
regeneran ejecutando los notebooks en orden y no se versionan por separado.

## Cómo se crearon las referencias

```bash
dvc init
dvc add harmonie_crudo.csv observaciones_horarias.csv dataset_modelado.csv
git add harmonie_crudo.csv.dvc observaciones_horarias.csv.dvc dataset_modelado.csv.dvc \
        .dvc/config .dvcignore .gitignore
```

## Comprobar que los datos locales corresponden a esta versión del código

Con los datos en la raíz del proyecto:

```bash
dvc status
```

Si algún fichero no coincide con el hash de su `.dvc`, DVC lo indica como modificado. El
almacenamiento remoto de DVC es privado, por lo que `dvc pull` solo funciona con acceso a él.

## Relación con MLflow

DVC fija la versión de los datos y MLflow (notebook 8) registra los experimentos y el modelo
desplegado. El commit de git enlaza ambos: el mismo commit contiene los `.dvc` con los hashes
de los datos y el código que generó las métricas registradas.
