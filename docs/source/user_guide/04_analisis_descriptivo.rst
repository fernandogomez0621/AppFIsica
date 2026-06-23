================================
4. Análisis Descriptivo
================================

Esta guía te enseñará a realizar análisis exploratorio completo de tus datos de crédito hipotecario, incluyendo estadísticas univariadas, bivariadas y visualizaciones interactivas.

Objetivo del Módulo
===================

El módulo de análisis descriptivo te permite:

* 📊 **Explorar distribuciones** de variables individuales
* 📈 **Analizar correlaciones** entre variables
* 🔍 **Detectar outliers** y valores atípicos
* 📉 **Visualizar patrones** con gráficos interactivos
* 📋 **Generar reportes** estadísticos completos
* 🎯 **Identificar insights** para modelado

Acceso al Módulo
================

En el sidebar, click en:

.. code-block:: text

   📈 Análisis → 📈 Análisis Descriptivo

Análisis Univariado
====================

Selección de Variables
-----------------------

**Paso 1: Seleccionar variables**

.. code-block:: text

   Selecciona variables para analizar:
   
   ☑ edad
   ☑ salario_mensual
   ☑ puntaje_datacredito
   ☑ dti
   ☑ nivel_riesgo

**Paso 2: Ejecutar análisis**

.. code-block:: text

   [🎯 Analizar Variables Seleccionadas]

Estadísticas Descriptivas
--------------------------

Para cada variable numérica, obtendrás:

.. code-block:: text

   Variable: salario_mensual
   ═══════════════════════════
   
   Medidas de Tendencia Central:
   • Media:           $4,235,000
   • Mediana:         $3,850,000
   • Moda:            $3,500,000
   
   Medidas de Dispersión:
   • Desv. Estándar:  $2,150,000
   • Varianza:        4.62e+12
   • Rango:           $24,000,000
   • IQR:             $2,800,000
   
   Percentiles:
   • P5:              $1,500,000
   • P25:             $2,600,000
   • P50 (Mediana):   $3,850,000
   • P75:             $5,400,000
   • P95:             $8,200,000
   
   Forma de la Distribución:
   • Asimetría:       1.23 (sesgada a derecha)
   • Curtosis:        2.45 (leptocúrtica)
   
   Tests de Normalidad:
   • Shapiro-Wilk:    p-value = 0.001 (No normal)
   • Kolmogorov-S:    p-value = 0.003 (No normal)

Visualizaciones Univariadas
----------------------------

**1. Histograma con Curva de Densidad**

Muestra la distribución de frecuencias con curva KDE superpuesta.

**2. Boxplot (Diagrama de Cajas)**

Identifica:
- Mediana (línea central)
- Cuartiles Q1 y Q3 (caja)
- Valores atípicos (puntos fuera de bigotes)
- Rango intercuartílico

**3. Q-Q Plot (Gráfico Cuantil-Cuantil)**

Compara la distribución con una normal teórica.

**4. Violin Plot**

Combina boxplot con densidad de probabilidad.

.. tip::
   Usa el botón de descarga en cada gráfico para guardar las visualizaciones.

Variables Categóricas
----------------------

Para variables categóricas (ej: ``nivel_riesgo``):

.. code-block:: text

   Variable: nivel_riesgo
   ═══════════════════════
   
   Frecuencias:
   • Bajo:   6,074 (60.7%) ████████████████████████
   • Medio:  2,943 (29.4%) ████████████
   • Alto:     983 ( 9.8%) ████
   
   Moda: Bajo
   Entropía: 1.23 bits
   Índice de Gini: 0.54

**Visualizaciones:**

- Gráfico de barras
- Gráfico de pastel
- Tabla de frecuencias

Análisis Bivariado
==================

Correlaciones
-------------

**Matriz de Correlación:**

.. code-block:: text

   [🔗 Calcular Matriz de Correlación]

Genera una matriz de calor mostrando correlaciones entre todas las variables numéricas.

**Interpretación:**

.. code-block:: text

   Correlación    Interpretación
   ─────────────────────────────────
   0.8 - 1.0      Muy fuerte positiva
   0.6 - 0.8      Fuerte positiva
   0.4 - 0.6      Moderada positiva
   0.2 - 0.4      Débil positiva
   -0.2 - 0.2     Muy débil/ninguna
   -0.4 - -0.2    Débil negativa
   -0.6 - -0.4    Moderada negativa
   -0.8 - -0.6    Fuerte negativa
   -1.0 - -0.8    Muy fuerte negativa

**Correlaciones Importantes:**

.. code-block:: text

   Top 5 Correlaciones Positivas:
   1. salario_mensual ↔ patrimonio_total: 0.72
   2. valor_inmueble ↔ monto_credito: 0.89
   3. edad ↔ antiguedad_empleo: 0.58
   4. puntaje_datacredito ↔ capacidad_ahorro: 0.45
   5. estrato ↔ salario_mensual: 0.51
   
   Top 5 Correlaciones Negativas:
   1. dti ↔ capacidad_residual: -0.68
   2. numero_demandas ↔ puntaje_datacredito: -0.23
   3. ltv ↔ cuota_inicial: -0.85
   4. edad ↔ plazo_credito: -0.42
   5. egresos ↔ capacidad_ahorro: -0.78

Scatter Plots
-------------

**Gráficos de Dispersión:**

Selecciona dos variables para ver su relación:

.. code-block:: text

   Variable X: salario_mensual
   Variable Y: valor_inmueble
   Color por: nivel_riesgo
   
   [📊 Generar Scatter Plot]

**Características:**

- Puntos coloreados por categoría
- Línea de tendencia
- Coeficiente de correlación
- Intervalos de confianza

Análisis por Grupos
-------------------

**Comparar distribuciones por categoría:**

.. code-block:: text

   Variable numérica: dti
   Agrupar por: nivel_riesgo
   
   [📊 Comparar Grupos]

**Resultados:**

.. code-block:: text

   DTI por Nivel de Riesgo:
   
   Bajo:   Media = 23.5%, Mediana = 22.8%
   Medio:  Media = 29.2%, Mediana = 28.5%
   Alto:   Media = 34.8%, Mediana = 35.2%
   
   Test ANOVA: F = 245.3, p < 0.001
   Conclusión: Diferencias significativas entre grupos

**Visualizaciones:**

- Boxplots comparativos
- Violin plots por grupo
- Histogramas superpuestos

Tablas de Contingencia
-----------------------

Para dos variables categóricas:

.. code-block:: text

   Variable 1: tipo_empleo
   Variable 2: nivel_riesgo
   
   [📋 Generar Tabla de Contingencia]

**Resultado:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 
     - Bajo
     - Medio
     - Alto
   * - Formal
     - 4,250 (65%)
     - 1,580 (24%)
     - 720 (11%)
   * - Informal
     - 1,420 (51%)
     - 980 (35%)
     - 390 (14%)
   * - Independiente
     - 404 (62%)
     - 183 (28%)
     - 63 (10%)

**Test Chi-Cuadrado:**

.. code-block:: text

   χ² = 89.5, p < 0.001
   Conclusión: Asociación significativa

Detección de Outliers
=====================

Métodos Automáticos
-------------------

**1. Método IQR (Rango Intercuartílico)**

.. code-block:: text

   Outliers detectados en salario_mensual:
   
   Límite inferior: Q1 - 1.5*IQR = $0
   Límite superior: Q3 + 1.5*IQR = $9,600,000
   
   Outliers superiores: 23 valores
   Valores: [$10.2M, $11.5M, $12.8M, ...]

**2. Método Z-Score**

.. code-block:: text

   Outliers (|Z| > 3):
   
   salario_mensual: 15 valores extremos
   valor_inmueble: 8 valores extremos
   patrimonio_total: 12 valores extremos

**3. Isolation Forest**

Algoritmo de ML para detectar anomalías multivariadas.

Visualización de Outliers
--------------------------

- Boxplots con outliers marcados
- Scatter plots con outliers resaltados
- Histogramas con zonas de outliers

Reportes Estadísticos
=====================

Reporte Completo
----------------

.. code-block:: text

   [📄 Generar Reporte Completo]

Incluye:

1. **Resumen Ejecutivo**
   - Tamaño del dataset
   - Variables analizadas
   - Hallazgos principales

2. **Estadísticas Univariadas**
   - Todas las variables numéricas
   - Todas las variables categóricas

3. **Análisis Bivariado**
   - Matriz de correlación
   - Top correlaciones
   - Tests estadísticos

4. **Detección de Anomalías**
   - Outliers por variable
   - Registros problemáticos

5. **Recomendaciones**
   - Variables para transformar
   - Outliers a tratar
   - Próximos pasos

Exportar Resultados
-------------------

**Formatos disponibles:**

.. code-block:: text

   [📊 Exportar a Excel]
   [📄 Exportar a PDF]
   [📋 Exportar a CSV]
   [🖼️ Exportar Gráficos]

Casos de Uso
============

**Caso 1: Exploración Inicial**

Objetivo: Entender la estructura de los datos.

Pasos:
1. Analizar todas las variables numéricas
2. Revisar distribuciones
3. Identificar outliers
4. Generar reporte inicial

**Caso 2: Validación de Calidad**

Objetivo: Verificar calidad antes de modelar.

Pasos:
1. Detectar valores atípicos
2. Verificar normalidad
3. Analizar correlaciones
4. Identificar problemas

**Caso 3: Feature Selection**

Objetivo: Seleccionar variables para modelos.

Pasos:
1. Calcular correlaciones con target
2. Identificar multicolinealidad
3. Analizar importancia de variables
4. Seleccionar features óptimas

Tips y Mejores Prácticas
=========================

✅ **Haz:**

- Analiza todas las variables antes de modelar
- Documenta hallazgos importantes
- Compara distribuciones por grupos
- Verifica supuestos estadísticos

❌ **Evita:**

- Ignorar outliers sin investigar
- Asumir normalidad sin verificar
- Pasar por alto correlaciones altas
- Omitir análisis bivariado

Troubleshooting
===============

**Problema: Gráficos no se generan**

Solución: Verifica que hayas seleccionado variables válidas.

**Problema: Tests estadísticos fallan**

Solución: Asegúrate de tener suficientes datos (n > 30).

**Problema: Correlaciones inesperadas**

Solución: Verifica calidad de datos y outliers.

Próximos Pasos
==============

Con tu análisis completo:

1. **Crear características**: :doc:`05_ingenieria_caracteristicas`
2. **Segmentar clientes**: :doc:`06_clustering`
3. **Entrenar RBM**: :doc:`07_rbm`

¡Has completado el análisis exploratorio! 📊