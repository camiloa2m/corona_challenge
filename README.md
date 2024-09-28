# Digits Image Classification API

Esta es una API RESTful construida con FastAPI para clasificar imágenes utilizando un modelo de machine learning preentrenado. La API permite recibir solicitudes de predicción y devolver resultados de manera clara y precisa.

## Características

- Interfaz RESTful sencilla y clara.
- Predicciones rápidas para clasificaciones de imágenes de dígitos.
- Contenerización con Docker para un despliegue fácil y reproducible.

## Requisitos

- Python 3.7 o superior
- Docker (opcional, para contenerización)

## Instalación (Opción A)

### Instalar Dependencias

Puedes instalar las dependencias requeridas ejecutando:

```bash
pip install -r requirements.txt
```

### Ejecutar la API Localmente

Para ejecutar la API localmente, puedes usar el siguiente comando:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
La API estará disponible en http://localhost:8000.

## Contenerización con Docker  (Opción B)

### 1. Construir la Imagen

Para contenerizar la aplicación, asegúrate de tener Docker instalado. Luego, navega al directorio donde se encuentra tu Dockerfile y ejecuta el siguiente comando para construir la imagen Docker:

```bash
docker build -t nombre_de_tu_imagen .
```

**Ejemplo:** Si decides nombrar tu imagen digits_classifier, el comando sería:

```bash
docker build -t digits_classifier .
```

### 2. Lanzar el Contenedor

Una vez que la imagen se ha construido, puedes lanzar el contenedor utilizando el siguiente comando:

```bash
docker run --name nombre_del_contenedor -d -p 8000:8000 nombre_de_tu_imagen
```

**Ejemplo:** Si decides nombrar tu contenedor clf_digits_container, el comando sería:

```bash
docker run --name clf_digits_container -d -p 8000:8000 digits_classifier
```

* **--name nombre_del_contenedor:** Asigna un nombre a tu contenedor.
* **-d:** Ejecuta el contenedor en segundo plano (modo detached).
* **-p 8000:8000:** Mapea el puerto 8000 del contenedor al puerto 8000 de tu máquina host.

### 3. Acceder a la API

Una vez que el contenedor esté en ejecución, puedes acceder a la API en:

```
http://localhost:8000
```

* Para ver la documentación automática generada por FastAPI, visita:
    ```bash
    http://localhost:8000/docs
    ```

* Endpoints Disponibles:  `/predict` método: POST
    ```bash
    http://localhost:8000/predict

    ```

    Este endpoint recibe una imagen en base64 (string), siguiendo el schema JSON siguiente, y devuelve la clase predicha.

    *Ejemplo de Solicitud:*
    ```json
    {
        "subject": {
            "type": "base64",
            "value": "aquí_va_codificación_base64_de_la_imagen_como_string"
        },
        "subject_type": "Image"
    }
    ```
    **Nota: Solo es necesario modificar `value`.**

    *Respuesta:*
    ```json
    {
        "prediction": 1
    }
    ```

## Estimación de Latencia

La latencia de la API dependerá de varios factores, incluidos el tamaño de la imagen, la capacidad de computo de la máquina que corre el modelo y la carga del servidor. En condiciones óptimas, se espera una latencia de 9 ms por request.





