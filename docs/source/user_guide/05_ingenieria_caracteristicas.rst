====================================
5. Ingeniería de Características
====================================

Esta guía te enseñará a crear variables derivadas y transformaciones que mejoren el poder predictivo de tus modelos de Machine Learning.

Objetivo del Módulo
===================

El módulo de ingeniería de características te permite:

* 🔧 **Crear variables derivadas** automáticamente
* 📐 **Transformar variables** (log, sqrt, polinomiales)
* 🎯 **Codificar categóricas** (one-hot, label encoding)
* 📊 **Normalizar y escalar** datos
* 🔗 **Generar interacciones** entre variables
* 💡 **Aplicar domain knowledge** financiero

Acceso al Módulo
================

En el sidebar, click en:

.. code-block:: text

   📈 Análisis → 🔧 Ingeniería de Características

Variables Derivadas Automáticas
================================

El sistema genera automáticamente 15+ características derivadas:

Ratios Financieros
------------------

.. code-block:: text

   ✓ dti (Debt-to-Income): cuota / salario * 100
   ✓ ltv (Loan-to-Value): monto / valor_inmueble * 100
   ✓ ratio_patrimonio_deuda: patrimonio / monto
   ✓ ratio_cuota_ahorro: cuota / capacidad_ahorro
   ✓ ratio_egreso_salario: egresos / salario * 100

Capacidades
-----------

.. code-block:: text

   ✓ capacidad_ahorro: salario - egresos
   ✓ capacidad_residual: capacidad_ahorro - cuota
   ✓ meses_colchon: saldo_banco / cuota

Scores
------

.. code-block:: text

   ✓ score_edad: Puntaje basado en edad óptima
   ✓ score_estabilidad_laboral: Antigüedad + tipo empleo
   ✓ riesgo_legal: Basado en número de demandas

Interacciones
-------------

.. code-block:: text

   ✓ educacion_x_salario: nivel_educacion * salario
   ✓ edad_x_antiguedad: edad * antiguedad_empleo
   ✓ ltv_x_puntaje: ltv * (900 - puntaje_datacredito)

Transformaciones
================

Transformaciones Numéricas
---------------------------

**1. Logarítmica**

Para variables con distribución sesgada:

.. code-block:: text

   Variables a transformar:
   ☑ salario_mensual
   ☑ valor_inmueble
   ☑ patrimonio_total
   
   Transformación: log(x + 1)
   
   [🔄 Aplicar Transformación Log]

**Resultado:**

.. code-block:: text

   salario_mensual_log creada
   Asimetría antes: 1.23 → después: 0.15
   Distribución más normal ✓

**2. Raíz Cuadrada**

Para reducir impacto de outliers:

.. code-block:: text

   Variables: monto_credito, cuota_mensual
   Transformación: sqrt(x)

**3. Box-Cox**

Transformación automática óptima:

.. code-block:: text

   [🎯 Aplicar Box-Cox Automático]
   
   Lambda óptimo encontrado: 0.23
   Transformación aplicada ✓

Transformaciones Polinomiales
------------------------------

Crear términos cuadráticos y cúbicos:

.. code-block:: text

   Variable: edad
   
   ☑ edad²
   ☑ edad³
   
   [📐 Generar Polinomios]

**Uso:** Capturar relaciones no lineales.

Codificación de Variables
==========================

Variables Categóricas
---------------------

**1. One-Hot Encoding**

.. code-block:: text

   Variable: tipo_empleo
   
   Resultado:
   ✓ tipo_empleo_Formal
   ✓ tipo_empleo_Informal
   ✓ tipo_empleo_Independiente

**2. Label Encoding**

.. code-block:: text

   Variable: nivel_educacion
   
   Mapeo:
   Bachiller → 1
   Técnico → 2
   Profesional → 3
   Posgrado → 4

**3. Target Encoding**

Codifica basado en la media del target:

.. code-block:: text

   Variable: ciudad
   Target: nivel_riesgo
   
   Bogotá → 0.58 (% riesgo alto)
   Medellín → 0.62
   Cali → 0.65
   ...

Variables Ordinales
-------------------

Para variables con orden natural:

.. code-block:: text

   Variable: estrato_socioeconomico
   Ya es ordinal: 1, 2, 3, 4, 5, 6 ✓
   
   Variable: nivel_dti
   Mapeo:
   Excelente → 1
   Bueno → 2
   Aceptable → 3
   Límite → 4
   Alto → 5
   Crítico → 6

Normalización y Escalado
=========================

Métodos de Escalado
-------------------

**1. StandardScaler (Z-Score)**

.. code-block:: text

   Fórmula: (x - μ) / σ
   
   Resultado: Media = 0, Desv. Est. = 1
   
   Uso: Algoritmos sensibles a escala (SVM, KNN, RBM)

**2. MinMaxScaler**

.. code-block:: text

   Fórmula: (x - min) / (max - min)
   
   Resultado: Rango [0, 1]
   
   Uso: Redes neuronales, algoritmos de distancia

**3. RobustScaler**

.. code-block:: text

   Fórmula: (x - mediana) / IQR
   
   Resultado: Robusto a outliers
   
   Uso: Datos con muchos outliers

Aplicar Escalado
----------------

.. code-block:: text

   Método de escalado: StandardScaler
   
   Variables a escalar:
   ☑ salario_mensual
   ☑ valor_inmueble
   ☑ puntaje_datacredito
   ☑ dti
   ☑ ltv
   
   [⚖️ Aplicar Escalado]

.. warning::
   Guarda los parámetros del scaler para aplicar a datos nuevos.

Binning y Discretización
=========================

Crear Rangos
------------

**Binning Equidistante:**

.. code-block:: text

   Variable: edad
   Número de bins: 4
   
   Resultado:
   [22-32): Joven
   [32-42): Adulto Joven
   [42-52): Adulto
   [52-65]: Adulto Mayor

**Binning por Cuantiles:**

.. code-block:: text

   Variable: salario_mensual
   Cuantiles: 5 (quintiles)
   
   Resultado:
   Q1: Muy Bajo ($1M - $2M)
   Q2: Bajo ($2M - $3.5M)
   Q3: Medio ($3.5M - $5M)
   Q4: Medio-Alto ($5M - $8M)
   Q5: Alto ($8M+)

**Binning Personalizado:**

.. code-block:: text

   Variable: puntaje_datacredito
   
   Rangos personalizados:
   [150-400): Crítico
   [400-500): Muy Malo
   [500-600): Malo
   [600-700): Regular
   [700-800): Bueno
   [800-850): Muy Bueno
   [850-950]: Excelente

Selección de Características
=============================

Métodos Automáticos
-------------------

**1. Correlación con Target**

.. code-block:: text

   [🎯 Calcular Importancia por Correlación]
   
   Top 10 Features:
   1. dti: 0.68
   2. puntaje_datacredito: -0.54
   3. capacidad_residual: -0.48
   4. ltv: 0.42
   5. score_estabilidad: -0.38
   ...

**2. Mutual Information**

Captura relaciones no lineales:

.. code-block:: text

   [📊 Calcular Mutual Information]
   
   Top features por MI:
   1. dti: 0.42
   2. capacidad_residual: 0.38
   3. puntaje_datacredito: 0.35
   ...

**3. Feature Importance (Tree-based)**

Usa Random Forest para calcular importancia:

.. code-block:: text

   [🌳 Calcular Feature Importance]
   
   Importancia por Random Forest:
   1. dti: 0.18
   2. puntaje_datacredito: 0.15
   3. capacidad_residual: 0.12
   ...

Eliminar Features Redundantes
------------------------------

**Multicolinealidad:**

.. code-block:: text

   [🔍 Detectar Multicolinealidad]
   
   VIF (Variance Inflation Factor):
   
   salario_mensual: 2.3 ✓
   patrimonio_total: 8.5 ⚠️ (alta correlación)
   capacidad_ahorro: 12.3 ⚠️ (alta correlación)
   
   Recomendación: Eliminar capacidad_ahorro

**Correlación Alta:**

.. code-block:: text

   Pares con correlación > 0.9:
   
   valor_inmueble ↔ monto_credito: 0.95
   Acción: Mantener solo monto_credito

Casos de Uso
============

**Caso 1: Preparar para RBM**

Objetivo: Crear features para entrenamiento de RBM.

Pasos:
1. Generar variables derivadas automáticas
2. Normalizar con StandardScaler
3. Seleccionar top 20 features
4. Guardar transformaciones

**Caso 2: Mejorar Modelo Supervisado**

Objetivo: Aumentar precisión de clasificadores.

Pasos:
1. Crear interacciones polinomiales
2. Aplicar transformaciones log
3. Codificar categóricas (one-hot)
4. Seleccionar por importancia

**Caso 3: Reducir Dimensionalidad**

Objetivo: Eliminar features redundantes.

Pasos:
1. Calcular VIF
2. Eliminar alta multicolinealidad
3. Seleccionar por mutual information
4. Validar con modelo baseline

Tips y Mejores Prácticas
=========================

✅ **Haz:**

- Crea features basadas en domain knowledge
- Normaliza antes de RBM y redes neuronales
- Guarda transformaciones para producción
- Valida features con modelos simples

❌ **Evita:**

- Crear demasiadas features (curse of dimensionality)
- Aplicar transformaciones sin entender el impacto
- Olvidar escalar datos de test
- Usar target encoding sin validación cruzada

Guardar Transformaciones
========================

.. code-block:: text

   [💾 Guardar Pipeline de Transformaciones]
   
   Guardado:
   ✓ Scaler (StandardScaler)
   ✓ Encoders (OneHot, Label)
   ✓ Feature names
   ✓ Parámetros de transformación
   
   Archivo: feature_pipeline.pkl

.. important::
   Usa el mismo pipeline en producción para consistencia.

Troubleshooting
===============

**Problema: Transformación log falla**

Solución: Verifica que no haya valores negativos o cero.

**Problema: Scaler produce NaN**

Solución: Elimina valores faltantes antes de escalar.

**Problema: Demasiadas features después de one-hot**

Solución: Usa target encoding o agrupa categorías raras.

Próximos Pasos
==============

Con tus features creadas:

1. **Segmentar**: :doc:`06_clustering`
2. **Entrenar RBM**: :doc:`07_rbm`
3. **Modelos supervisados**: :doc:`08_modelos_supervisados`

¡Features listas para modelado! 🔧