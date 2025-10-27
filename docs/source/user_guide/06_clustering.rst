============================
6. Clustering y Segmentación
============================

Esta guía te enseñará a segmentar clientes en grupos homogéneos usando algoritmos de clustering no supervisado.

Objetivo del Módulo
===================

El módulo de clustering te permite:

* 🎯 **Segmentar clientes** en grupos similares
* 📊 **Identificar patrones** ocultos en datos
* 🔍 **Descubrir perfiles** de riesgo
* 📈 **Optimizar estrategias** por segmento
* 🎨 **Visualizar clusters** en 2D/3D
* 💡 **Generar insights** accionables

Acceso al Módulo
================

En el sidebar, click en:

.. code-block:: text

   📈 Análisis → 🎯 Clustering

Algoritmos Disponibles
=======================

K-Means
-------

**Características:**

* Rápido y escalable
* Requiere especificar número de clusters
* Sensible a outliers
* Clusters esféricos

**Cuándo usar:** Datasets grandes, clusters bien separados.

DBSCAN
------

**Características:**

* No requiere número de clusters
* Detecta outliers automáticamente
* Clusters de forma arbitraria
* Sensible a parámetros

**Cuándo usar:** Clusters de forma irregular, presencia de ruido.

Hierarchical Clustering
------------------------

**Características:**

* Crea dendrograma
* No requiere número de clusters a priori
* Computacionalmente costoso
* Visualización intuitiva

**Cuándo usar:** Datasets pequeños, análisis exploratorio.

Gaussian Mixture Models (GMM)
------------------------------

**Características:**

* Clusters probabilísticos
* Flexible (elípticos)
* Asigna probabilidades
* Más lento que K-Means

**Cuándo usar:** Clusters superpuestos, incertidumbre en asignación.

Proceso de Clustering
=====================

Paso 1: Seleccionar Variables
------------------------------

.. code-block:: text

   Variables para clustering:
   
   ☑ puntaje_datacredito
   ☑ dti
   ☑ capacidad_residual
   ☑ ltv
   ☑ salario_mensual
   ☑ patrimonio_total

.. tip::
   Selecciona variables relevantes y no correlacionadas.

Paso 2: Preprocesamiento
-------------------------

.. code-block:: text

   [⚙️ Preprocesar Datos]
   
   ✓ Normalización (StandardScaler)
   ✓ Eliminación de outliers extremos
   ✓ Reducción de dimensionalidad (PCA)

Paso 3: Determinar Número Óptimo
---------------------------------

**Método del Codo (Elbow Method):**

.. code-block:: text

   [📊 Calcular Método del Codo]
   
   K=2: Inertia = 1250.5
   K=3: Inertia = 890.2  ← Codo
   K=4: Inertia = 750.8
   K=5: Inertia = 680.3
   
   Número óptimo sugerido: 3

**Silhouette Score:**

.. code-block:: text

   K=2: Score = 0.65
   K=3: Score = 0.72  ← Mejor
   K=4: Score = 0.68
   K=5: Score = 0.61

Paso 4: Entrenar Modelo
------------------------

.. code-block:: text

   Algoritmo: K-Means
   Número de clusters: 3
   
   [🎯 Entrenar Clustering]
   
   ✓ Modelo entrenado
   ✓ Clusters asignados
   ✓ Centroides calculados

Análisis de Clusters
=====================

Perfil de Clusters
------------------

**Cluster 0: Bajo Riesgo (60%)**

.. code-block:: text

   Características:
   • Puntaje DataCrédito: 750-850
   • DTI: 15-25%
   • Capacidad residual: Alta
   • Salario: Medio-Alto
   • Patrimonio: Alto
   
   Perfil: Clientes premium, bajo riesgo

**Cluster 1: Riesgo Moderado (25%)**

.. code-block:: text

   Características:
   • Puntaje DataCrédito: 600-750
   • DTI: 25-35%
   • Capacidad residual: Media
   • Salario: Medio
   • Patrimonio: Medio
   
   Perfil: Clientes estándar, riesgo controlado

**Cluster 2: Alto Riesgo (15%)**

.. code-block:: text

   Características:
   • Puntaje DataCrédito: 350-600
   • DTI: 35-45%
   • Capacidad residual: Baja
   • Salario: Bajo
   • Patrimonio: Bajo
   
   Perfil: Clientes de alto riesgo

Visualizaciones
===============

Scatter Plot 2D
---------------

Proyección PCA en 2 dimensiones:

.. code-block:: text

   [📊 Visualizar Clusters 2D]
   
   Componente 1 vs Componente 2
   Colores por cluster
   Centroides marcados

Scatter Plot 3D
---------------

Proyección en 3 dimensiones:

.. code-block:: text

   [🎨 Visualizar Clusters 3D]
   
   Interactivo con Plotly
   Rotación y zoom

Dendrograma
-----------

Para clustering jerárquico:

.. code-block:: text

   [🌳 Generar Dendrograma]
   
   Muestra jerarquía de agrupación
   Ayuda a determinar número de clusters

Casos de Uso
============

**Caso 1: Segmentación de Mercado**

Objetivo: Identificar grupos de clientes para estrategias diferenciadas.

**Caso 2: Detección de Anomalías**

Objetivo: Identificar clientes atípicos usando DBSCAN.

**Caso 3: Personalización de Productos**

Objetivo: Diseñar productos específicos por segmento.

Tips y Mejores Prácticas
=========================

✅ **Haz:**

- Normaliza datos antes de clustering
- Prueba múltiples algoritmos
- Valida clusters con métricas
- Interpreta resultados con domain knowledge

❌ **Evita:**

- Usar demasiadas variables
- Ignorar outliers
- Forzar número de clusters sin validar
- Clustering sin preprocesamiento

Troubleshooting
===============

**Problema: Clusters no tienen sentido**

Solución: Revisa selección de variables y normalización.

**Problema: Todos los puntos en un cluster**

Solución: Ajusta parámetros o prueba otro algoritmo.

Próximos Pasos
==============

Con tus clusters definidos:

1. **Entrenar RBM**: :doc:`07_rbm`
2. **Modelos por cluster**: :doc:`08_modelos_supervisados`
3. **Predicción segmentada**: :doc:`09_prediccion`

¡Segmentación completada! 🎯