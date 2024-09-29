# Propuesta: Manejo de Drift y Re-entrenamiento

## 1. Detección de Drift

Para detectar el drift en los datos, podemos centrarnos tanto en los datos de entrada como en las predicciones del modelo.

### 1.1. Detección de Drift en Datos de Entrada:

#### A. Métricas de Distancia Estadística

* **Prueba de Kolmogorov-Smirnov (KS):** Para detectar cambios en la distribución de características continuas.

* **Índice de Estabilidad Poblacional (PSI):** Mide el cambio en la distribución de datos de entrada a lo largo del tiempo, comparando nuevos datos con los datos base de entrenamiento.

* **Kullback-Leibler Divergence (KL Divergence):** Mide cómo una distribución de probabilidad se desvía de una segunda distribución de probabilidad esperada.

* **Jensen-Shannon Divergence (JS Divergence):** Es una versión simétrica de la divergencia KL, lo que la hace adecuada para medir la similitud entre dos distribuciones de probabilidad.


#### B. Detección de Cambio de Covariables
Entrenar un clasificador (por ejemplo, regresión logística) para distinguir entre datos de entrenamiento y nuevos datos. Si el clasificador puede diferenciar fácilmente, podría haber un drift.

### 1.2. Detección de Drift en Predicciones:

#### A. Monitoreo de Precisión del Modelo

Realizar un seguimiento de las métricas de precisión del modelo a lo largo del tiempo utilizando un conjunto de retención o datos de retroalimentación.

* Alerta basada en umbrales: Cuando la precisión cae por debajo de un cierto umbral, marca para detectar drift.

#### B. Detección de Concept Drift

Monitorear la distribución de predicciones y analizar la confianza en las predicciones. Un cambio repentino en la distribución o una disminución en la confianza podrían indicar drift.

#### C. Métricas y Métodos

* PSI para distribuciones de datos de entrada.
* Precisión y F1-score para seguir el rendimiento de las predicciones.
* Distribuciones de errores (por ejemplo, analizando los residuos de valores verdaderos frente a los predichos).

## 2. Estrategia de Re-entrenamiento

Una vez que se detecte el drift, el sistema debe decidir cuándo y cómo reentrenar el modelo. La decisión dependerá de varios factores:

### 2.1.  Umbrales para el Re-entrenamiento

* Establecer umbrales de reentrenamiento basados en métricas de drift de datos (por ejemplo, PSI > 0.2).
* Monitorear el rendimiento del modelo en datos de retención o en datos del mundo real a lo largo del tiempo. Si se observa una caída significativa en el rendimiento (por ejemplo, caída del 10% en la precisión), activar el reentrenamiento.

### 2.2. Frecuencia de Re-entrenamiento

* **Reentrenamiento Programado:** Tener un calendario regular (por ejemplo, mensual o trimestral).
* **Reentrenamiento Activado:** Iniciar el reentrenamiento cuando se detecte un drift significativo.

### 2.3. Validación del Modelo

* Antes de la implementación, validar el modelo reentrenado utilizando validación cruzada entre el nuevo conjunto de datos como en una mezcla del conjunto de datos antiguo.
* Comparar el rendimiento del modelo recién entrenado con el modelo en producción actual para asegurar mejoras.

### 2.4. Costos y Recursos

* **Costo computacional:** El reentrenamiento debe equilibrarse con los recursos computacionales y de tiempo disponibles.
* **Intervención Humana:** Si el reentrenamiento es costoso, involucrar la supervisión humana para la aprobación final de implementación.

## 3. Pipeline Automatizado para el Re-entrenamiento

Podemos diseñar un pipeline de reentrenamiento completamente automatizado con los siguientes componentes:

### 3.1. Recolección de Datos:

* **Monitoreo en tiempo real:** Recolectar y almacenar continuamente nuevos datos entrantes (por ejemplo, a través de plataformas de streaming como Kafka o mediante ingesta de datos por lotes).
* **Etiquetado de Datos:** Aprovechar los bucles de retroalimentación existentes o etiquetar nuevos datos a través de técnicas de supervisión débil para construir la verdad fundamental para el reentrenamiento.

### 3.2. Módulo de Detección de Drift:

* Ejecutar continuamente algoritmos de detección de drift entre los datos de entrada como en las predicciones.
* Utilizar mecanismos de alerta para notificar sobre el drift cuando los umbrales de detección se superen.

### 3.3. Flujo de Trabajo de Re-entrenamiento:
Reentrenamiento Automatizado del Modelo: Cuando se detecte drift, el sistema automáticamente:

* Extrae los datos más recientes (incluyendo el conjunto de datos antiguo).
* Activa la pipeline de entrenamiento del modelo.
* Valida el modelo (validación cruzada en datos nuevos y antiguos).

### 3.4. Despliegue de la Nueva Versión del Modelo:
* Utilizar **A/B Test** o **Canary Deployment** para implementar gradualmente el nuevo modelo sin interrumpir el servicio actual.
* Usar **Shadow Deployment** ejecutando el nuevo modelo en paralelo al viejo, y comparar las salidas sin afectar a los usuarios.
* Una vez que el nuevo modelo pase la validación, cambiar automáticamente el tráfico al nuevo modelo.

## Desafíos y Soluciones:

- ### Disponibilidad de Datos
    Puede que los nuevos datos no estén etiquetados. Para superar esto, se pueden implementar estrategias de aprendizaje semisupervisado o aprendizaje activo para solicitar etiquetas para un subconjunto de datos.

- ### Degradación del Modelo Durante el Reentrenamiento
    Asegurarse de que el modelo reentrenado supere o al menos iguale el rendimiento del modelo antiguo utilizando técnicas de evaluación robustas antes de la implementación.

- ### Carga Computacional
    Utilizar infraestructura en la nube para aumentar recursos cuando sea necesario para el reentrenamiento y apagarlos posteriormente para minimizar costos.

- ### Versionado y Monitoreo
    * Implementar versionado de modelos (por ejemplo, utilizando DVC o MLflow) para rastrear diferentes modelos, conjuntos de datos y rendimiento a lo largo del tiempo.
    * Monitorear continuamente el nuevo modelo después de la implementación para detectar anomalías o problemas.
