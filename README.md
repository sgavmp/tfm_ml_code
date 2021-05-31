# Código del TFM en colaboración con Darwinex

**Autor:** Sergio García Alonso

## Contenido
* datasets: Contiene los archivos con los datos para ejecutar las tareas de entrenamiento.
* data_profiling: Contiene los HTML de los estudios realizados sobre los archivos de datos utilizando la herramienta Pandas Profiling.
* notebooks: Contiene los notebooks para realizar las pruebas y el entrenamiento de los modelos, incluyendo su despliegue a S3.
* sagemaker: Contiene el Docker para ejecutar los scripts con Pycaret en Sagemaker, y los diferentes scripts para automatizar la creación de las tareas de entrenamiento y los punto de enlaces para cada modelo.

## Notas para ejecutar los Notebooks
Por defecto, los Notebooks leen los datos de S3, y almacena los modelos resultantes en S3. En cada notebook se encuentra comentado el código para leer los datos de manera local y almacenar los modelos tambíen en local.

## Sagemaker
