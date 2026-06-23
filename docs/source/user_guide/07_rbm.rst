===============================================
7. Máquina de Boltzmann Restringida (RBM)
===============================================

Esta guía te enseñará a entrenar y usar Máquinas de Boltzmann Restringidas para extracción de características latentes en datos de crédito hipotecario.

Objetivo del Módulo
===================

El módulo de RBM te permite:

* ⚡ **Entrenar RBM** desde cero
* 🧠 **Extraer características latentes** no lineales
* 📊 **Visualizar pesos** y activaciones
* 📈 **Monitorear convergencia** durante entrenamiento
* 💾 **Guardar modelos** entrenados
* 🔄 **Usar features** en modelos supervisados

¿Qué es una RBM?
================

Definición
----------

Una **Máquina de Boltzmann Restringida** es un modelo generativo no supervisado que:

* Aprende representaciones latentes de los datos
* Usa una arquitectura de dos capas (visible + oculta)
* Se entrena con **Contrastive Divergence (CD-k)**
* Extrae características útiles para modelos supervisados

Arquitectura
------------

.. code-block:: text

   Capa Oculta (h)
   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○  (100 unidades)
   ↕ ↕ ↕ ↕ ↕ ↕ ↕ ↕ ↕ ↕
   ● ● ● ● ● ● ● ● ● ●  (n_features unidades)
   Capa Visible (v)

**Características:**

* Sin conexiones dentro de cada capa (restricción)
* Conexiones completas entre capas
* Pesos bidireccionales W
* Sesgos a (visible) y b (oculto)

Función de Energía
------------------

.. math::

   E(v,h) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i W_{ij} h_j

Donde:

* :math:`v_i` son las unidades visibles
* :math:`h_j` son las unidades ocultas  
* :math:`W_{ij}` son los pesos de conexión
* :math:`a_i, b_j` son los sesgos

Acceso al Módulo
================

En el sidebar, click en:

.. code-block:: text

   🤖 Modelado → ⚡ Máquina de Boltzmann (RBM)

Configuración de Hiperparámetros
=================================

Parámetros Principales
----------------------

**1. Número de Unidades Ocultas**

.. code-block:: text

   Unidades ocultas: [100]
   Rango: 50 - 500

**¿Qué significa?**

El número de características latentes que la RBM aprenderá.

**Recomendaciones:**

* **50-100**: Datasets pequeños (<5K registros)
* **100-200**: Datasets medianos (5K-20K)
* **200-500**: Datasets grandes (>20K)

.. tip::
   Comienza con 100 y ajusta según resultados.

**2. Learning Rate (Tasa de Aprendizaje)**

.. code-block:: text

   Learning rate: [0.01]
   Rango: 0.001 - 0.1

**¿Qué significa?**

Qué tan rápido el modelo aprende de los datos.

**Recomendaciones:**

* **0.001-0.005**: Aprendizaje lento pero estable
* **0.01**: Valor por defecto balanceado
* **0.05-0.1**: Aprendizaje rápido pero inestable

.. warning::
   Learning rate muy alto puede causar divergencia.

**3. Número de Épocas**

.. code-block:: text

   Épocas: [100]
   Rango: 50 - 500

**¿Qué significa?**

Cuántas veces el modelo ve todo el dataset.

**Recomendaciones:**

* **50-100**: Pruebas rápidas
* **100-200**: Entrenamiento estándar
* **200-500**: Entrenamiento exhaustivo

**4. Batch Size**

.. code-block:: text

   Batch size: [64]
   Rango: 32 - 256

**¿Qué significa?**

Cuántos ejemplos se procesan simultáneamente.

**Recomendaciones:**

* **32-64**: Datasets pequeños, más estable
* **64-128**: Valor estándar
* **128-256**: Datasets grandes, más rápido

**5. Pasos de Contrastive Divergence (k)**

.. code-block:: text

   CD-k steps: [1]
   Rango: 1 - 10

**¿Qué significa?**

Cuántos pasos de Gibbs sampling se ejecutan.

**Recomendaciones:**

* **k=1**: Más rápido, suficiente para mayoría de casos
* **k=5**: Más preciso pero más lento
* **k=10**: Solo para investigación

Proceso de Entrenamiento
=========================

Paso 1: Preparar Datos
-----------------------

.. code-block:: text

   [⚙️ Preparar Datos para RBM]
   
   ✓ Normalización aplicada (StandardScaler)
   ✓ Variables seleccionadas: 15
   ✓ Registros de entrenamiento: 8,000
   ✓ Registros de validación: 2,000

Paso 2: Configurar Hiperparámetros
-----------------------------------

.. code-block:: text

   Configuración:
   • Unidades ocultas: 100
   • Learning rate: 0.01
   • Épocas: 100
   • Batch size: 64
   • CD-k: 1

Paso 3: Entrenar RBM
---------------------

.. code-block:: text

   [🎯 Entrenar RBM]
   
   Época 1/100: Loss = 125.3
   Época 10/100: Loss = 89.2
   Época 20/100: Loss = 67.5
   Época 30/100: Loss = 54.8
   ...
   Época 100/100: Loss = 23.1
   
   ✓ Entrenamiento completado
   Tiempo total: 2m 15s

Monitoreo del Entrenamiento
============================

Curva de Aprendizaje
---------------------

Gráfico que muestra la evolución del error:

.. code-block:: text

   Loss vs Épocas
   
   150 ┤
       │ ●
   100 ┤  ●●
       │    ●●●
    50 ┤       ●●●●●●●●●●●●●●●
       │
     0 └─────────────────────────
       0    25    50    75   100
            Épocas

**Interpretación:**

* Descenso rápido inicial: Aprendizaje efectivo
* Estabilización: Convergencia alcanzada
* Oscilaciones: Posible learning rate alto

Reconstrucción de Datos
------------------------

Compara datos originales vs reconstruidos:

.. code-block:: text

   Error de Reconstrucción:
   
   Media: 0.023
   Mediana: 0.019
   Máximo: 0.145
   
   Calidad: Excelente ✓

Visualización de Pesos
======================

Matriz de Pesos
---------------

.. code-block:: text

   [🎨 Visualizar Matriz de Pesos]
   
   Heatmap 15x100
   • Filas: Features de entrada
   • Columnas: Unidades ocultas
   • Color: Magnitud del peso

**Interpretación:**

* Pesos altos (rojo): Conexión fuerte
* Pesos bajos (azul): Conexión débil
* Patrones: Grupos de features relacionadas

Filtros Aprendidos
------------------

Visualiza qué aprende cada unidad oculta:

.. code-block:: text

   Unidad Oculta 1:
   Detecta: Alto salario + Bajo DTI
   
   Unidad Oculta 2:
   Detecta: Buen puntaje + Alta capacidad residual
   
   Unidad Oculta 3:
   Detecta: Patrimonio alto + Múltiples propiedades

Extracción de Características
==============================

Transformar Datos
-----------------

.. code-block:: text

   [🔄 Extraer Features con RBM]
   
   Datos originales: 10,000 × 15
   Features RBM: 10,000 × 100
   
   ✓ Transformación completada

**Uso de Features:**

Las 100 características latentes pueden usarse como entrada para:

* Modelos supervisados (Random Forest, XGBoost, etc.)
* Clustering mejorado
* Reducción de dimensionalidad

Análisis de Activaciones
-------------------------

.. code-block:: text

   Estadísticas de Activaciones:
   
   Media: 0.45
   Desv. Est.: 0.28
   Sparsity: 35% (unidades inactivas)
   
   Distribución: Balanceada ✓

Casos de Uso
============

Caso 1: Pre-entrenamiento para Deep Learning
---------------------------------------------

**Objetivo**: Inicializar pesos de red neuronal profunda.

**Pasos:**

1. Entrenar RBM en datos no etiquetados
2. Usar pesos como inicialización
3. Fine-tuning con backpropagation
4. Mejor convergencia y generalización

Caso 2: Feature Engineering Automático
---------------------------------------

**Objetivo**: Crear features no lineales automáticamente.

**Pasos:**

1. Entrenar RBM con 200 unidades ocultas
2. Extraer activaciones como nuevas features
3. Combinar con features originales
4. Entrenar Random Forest
5. Mejora de 3-5% en precisión

Caso 3: Detección de Anomalías
-------------------------------

**Objetivo**: Identificar solicitudes atípicas.

**Pasos:**

1. Entrenar RBM en datos normales
2. Calcular error de reconstrucción
3. Umbral: error > percentil 95
4. Marcar como anomalías

Caso 4: Reducción de Dimensionalidad
-------------------------------------

**Objetivo**: Reducir de 47 a 20 features.

**Pasos:**

1. Entrenar RBM con 20 unidades ocultas
2. Extraer activaciones
3. Usar en lugar de features originales
4. Mantener 90% de información

Tips y Mejores Prácticas
=========================

Selección de Hiperparámetros
-----------------------------

✅ **Haz:**

* Comienza con valores por defecto
* Usa validación cruzada
* Monitorea curva de aprendizaje
* Ajusta learning rate si no converge

❌ **Evita:**

* Learning rate muy alto (>0.1)
* Demasiadas unidades ocultas (overfitting)
* Muy pocas épocas (<50)
* Ignorar normalización de datos

Optimización del Entrenamiento
-------------------------------

**Para acelerar:**

* Aumenta batch size (128-256)
* Reduce épocas inicialmente
* Usa CD-1 en lugar de CD-5
* Entrena en GPU si disponible

**Para mejorar calidad:**

* Aumenta épocas (200-300)
* Reduce learning rate (0.005)
* Usa CD-5 o CD-10
* Aumenta unidades ocultas

Interpretación de Resultados
-----------------------------

**Señales de buen entrenamiento:**

* Loss decrece consistentemente
* Error de reconstrucción < 0.05
* Activaciones balanceadas (30-70%)
* Pesos no explotan ni desaparecen

**Señales de problemas:**

* Loss oscila o aumenta
* Error de reconstrucción > 0.1
* Todas las unidades activas/inactivas
* Pesos muy grandes (>10) o muy pequeños (<0.01)

Guardar y Cargar Modelos
=========================

Guardar Modelo
--------------

.. code-block:: text

   [💾 Guardar Modelo RBM]
   
   Guardado:
   ✓ Pesos W
   ✓ Sesgos a, b
   ✓ Hiperparámetros
   ✓ Scaler
   
   Archivo: rbm_model_20240115.pkl

Cargar Modelo
-------------

.. code-block:: text

   [📂 Cargar Modelo RBM]
   
   Selecciona archivo: rbm_model_20240115.pkl
   
   ✓ Modelo cargado
   ✓ Listo para transformar datos

Troubleshooting
===============

Problema 1: Loss no decrece
----------------------------

**Síntomas:**

.. code-block:: text

   Época 1: Loss = 125.3
   Época 50: Loss = 124.8
   Época 100: Loss = 124.5

**Causas posibles:**

* Learning rate muy bajo
* Datos no normalizados
* Inicialización pobre

**Soluciones:**

1. Aumenta learning rate a 0.05
2. Verifica normalización (media=0, std=1)
3. Reinicia con nueva semilla aleatoria

Problema 2: Loss explota
-------------------------

**Síntomas:**

.. code-block:: text

   Época 1: Loss = 125.3
   Época 5: Loss = 450.2
   Época 10: Loss = NaN

**Causas posibles:**

* Learning rate muy alto
* Gradientes explotan
* Datos con outliers extremos

**Soluciones:**

1. Reduce learning rate a 0.001
2. Usa gradient clipping
3. Elimina outliers extremos

Problema 3: Overfitting
-----------------------

**Síntomas:**

.. code-block:: text

   Train loss: 15.2
   Val loss: 45.8

**Soluciones:**

1. Reduce unidades ocultas
2. Agrega regularización L2
3. Usa dropout
4. Aumenta datos de entrenamiento

Problema 4: Entrenamiento muy lento
------------------------------------

**Soluciones:**

1. Aumenta batch size a 128
2. Reduce épocas a 50
3. Usa CD-1 en lugar de CD-5
4. Reduce número de features

Comparación con Otros Métodos
==============================

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Método
     - Velocidad
     - Calidad
     - Interpretabilidad
   * - RBM
     - Media
     - Alta
     - Media
   * - PCA
     - Rápida
     - Media
     - Alta
   * - Autoencoder
     - Lenta
     - Muy Alta
     - Baja
   * - t-SNE
     - Muy Lenta
     - Alta
     - Baja

Próximos Pasos
==============

Con tu RBM entrenada:

1. **Usar features**: :doc:`08_modelos_supervisados`
2. **Predecir riesgo**: :doc:`09_prediccion`
3. **Aprender más**: :doc:`11_rag_educativo`

¡RBM entrenada exitosamente! ⚡