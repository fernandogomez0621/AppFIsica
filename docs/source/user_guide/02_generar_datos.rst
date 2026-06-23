================================
2. Generar Datos Sintéticos
================================

Esta guía te enseñará a generar datasets sintéticos realistas de crédito hipotecario colombiano para entrenar y probar modelos de Machine Learning.

Objetivo del Módulo
===================

El generador de datos sintéticos te permite:

* 📊 **Crear datasets realistas** sin necesidad de datos reales
* 🎲 **Controlar la reproducibilidad** con semillas aleatorias
* 📈 **Ajustar el tamaño** del dataset (1,000 - 50,000 registros)
* 🎯 **Obtener distribuciones realistas** de riesgo crediticio
* 🔧 **Experimentar sin restricciones** de privacidad de datos
* 📚 **Aprender sobre variables** financieras y crediticias

¿Por Qué Usar Datos Sintéticos?
================================

Ventajas
--------

✅ **Sin restricciones de privacidad**
   Los datos son completamente artificiales, no hay información personal real.

✅ **Distribuciones controladas**
   Puedes generar exactamente la distribución de riesgo que necesitas.

✅ **Reproducibilidad perfecta**
   Usando la misma semilla, obtendrás exactamente los mismos datos.

✅ **Escalabilidad**
   Genera desde 1,000 hasta 50,000 registros en segundos.

✅ **Correlaciones realistas**
   Las variables mantienen relaciones lógicas del mundo real.

✅ **Ideal para aprendizaje**
   Experimenta sin riesgo de dañar datos reales.

Limitaciones
------------

⚠️ **No reemplazan datos reales**
   Para producción, siempre valida con datos reales.

⚠️ **Simplificaciones**
   Algunas relaciones complejas del mundo real están simplificadas.

⚠️ **Sesgos del generador**
   Los datos reflejan los supuestos programados en el generador.

Acceso al Módulo
================

**Paso 1: Navegar al módulo**

En el sidebar, click en:

.. code-block:: text

   📊 Gestión de Datos → 📊 Generar Datos

**Paso 2: Interfaz del generador**

Verás la pantalla principal con:

* Título: **"Generador de Datos Sintéticos"**
* Descripción del módulo
* Panel de configuración
* Botón de generación
* Área de resultados

Configuración del Generador
============================

Parámetros Principales
----------------------

**1. Número de Registros**

.. code-block:: text

   Número de registros a generar: [10000]
   Rango: 1,000 - 50,000

**¿Qué significa?**

El número total de solicitudes de crédito que se generarán.

**Recomendaciones:**

* **1,000 - 5,000**: Pruebas rápidas y exploración inicial
* **10,000**: Valor por defecto, balance entre velocidad y representatividad
* **20,000 - 30,000**: Entrenamiento de modelos robustos
* **50,000**: Datasets grandes para modelos complejos

.. tip::
   Para tu primer uso, mantén el valor por defecto de 10,000 registros.

**2. Semilla Aleatoria**

.. code-block:: text

   Semilla aleatoria: [42]
   Rango: 1 - 9999

**¿Qué significa?**

Un número que controla la generación aleatoria. La misma semilla produce exactamente los mismos datos.

**Casos de uso:**

* **Reproducibilidad**: Usa la misma semilla para obtener datos idénticos
* **Comparación**: Diferentes semillas generan datasets distintos
* **Debugging**: Semilla fija facilita encontrar problemas

.. note::
   La semilla por defecto es 42 (referencia a "La Guía del Autoestopista Galáctico").

**Ejemplo práctico:**

.. code-block:: python

   # Estos dos comandos generan EXACTAMENTE los mismos datos
   df1 = generar_datos(n=10000, semilla=42)
   df2 = generar_datos(n=10000, semilla=42)
   
   # Estos generan datos DIFERENTES
   df3 = generar_datos(n=10000, semilla=123)

Parámetros Avanzados
--------------------

.. note::
   Los parámetros avanzados están preconfigurados con valores óptimos. Solo modifícalos si tienes necesidades específicas.

**Expandir "Configuración Avanzada"** para ver:

**1. Distribución de Riesgo Objetivo**

.. code-block:: text

   Riesgo Bajo:  60% ━━━━━━━━━━━━━━━━━━━━━━━━
   Riesgo Medio: 25% ━━━━━━━━━━
   Riesgo Alto:  15% ━━━━━━

Esta es la distribución objetivo que el generador intentará alcanzar.

**2. Rangos de Variables**

Puedes ajustar los rangos de:

* Edad: 22-65 años (por defecto)
* Salario: Según educación y ciudad
* Puntaje DataCrédito: 350-850 (por defecto)
* Valor inmueble: Según ciudad y estrato

.. warning::
   Modificar estos rangos puede generar datos poco realistas. Solo hazlo si entiendes las implicaciones.

Proceso de Generación
======================

Paso a Paso
-----------

**Paso 1: Configurar parámetros**

1. Ajusta el número de registros (ej: 10,000)
2. Establece la semilla aleatoria (ej: 42)
3. Revisa configuración avanzada (opcional)

**Paso 2: Iniciar generación**

Click en el botón:

.. code-block:: text

   [🎯 Generar Dataset]

**Paso 3: Observar progreso**

Verás un spinner con el mensaje:

.. code-block:: text

   ⏳ Generando 10,000 registros...
   
   [FASE 1/6] Generando variables demográficas...
   ✓ Fase 1 completada
   
   [FASE 2/6] Generando variables laborales...
   ✓ Fase 2 completada
   
   [FASE 3/6] Generando variables financieras...
   ✓ Fase 3 completada
   
   [FASE 4/6] Generando variables del crédito...
   ✓ Fase 4 completada
   
   [FASE 5/6] Generando características derivadas...
   ✓ Fase 5 completada
   
   [FASE 6/6] Calculando nivel de riesgo REALISTA...
   ✓ Fase 6 completada

.. tip::
   La generación de 10,000 registros toma aproximadamente 5-10 segundos.

**Paso 4: Revisar resultados**

Una vez completado, verás:

* ✅ Mensaje de éxito
* 📊 Resumen estadístico
* 📈 Visualizaciones de distribución
* 💾 Opciones de descarga

Fases de Generación
-------------------

El generador trabaja en 6 fases secuenciales:

**Fase 1: Variables Demográficas**

Genera:

* ``edad``: Edad del solicitante (22-65 años)
* ``ciudad``: Ciudad de residencia (15 ciudades colombianas)
* ``estrato_socioeconomico``: Estrato 1-6
* ``nivel_educacion``: Bachiller, Técnico, Profesional, Posgrado
* ``estado_civil``: Soltero, Casado, Unión Libre, Divorciado, Viudo
* ``personas_a_cargo``: Número de dependientes (0-5)

**Fase 2: Variables Laborales**

Genera:

* ``tipo_empleo``: Formal, Informal, Independiente
* ``antiguedad_empleo``: Años en el empleo actual
* ``salario_mensual``: Ingreso mensual en COP
* ``egresos_mensuales``: Gastos mensuales totales

**Fase 3: Variables Financieras**

Genera:

* ``numero_demandas``: Demandas legales por dinero (0-3)
* ``puntaje_datacredito``: Score crediticio (350-850)
* ``numero_propiedades``: Propiedades que posee (0-3)
* ``patrimonio_total``: Patrimonio neto en COP
* ``saldo_promedio_banco``: Saldo promedio últimos 6 meses

**Fase 4: Variables del Crédito**

Genera:

* ``valor_inmueble``: Valor comercial de la propiedad
* ``anos_inmueble``: Antigüedad del inmueble
* ``porcentaje_cuota_inicial``: Porcentaje de cuota inicial (10-40%)
* ``valor_cuota_inicial``: Valor en COP de la cuota inicial
* ``monto_credito``: Monto solicitado del préstamo
* ``plazo_credito``: Plazo en años (10-30)
* ``tasa_interes_anual``: Tasa de interés anual (8.5-16%)
* ``cuota_mensual``: Cuota mensual del crédito
* ``ltv``: Loan-to-Value ratio (%)

**Fase 5: Características Derivadas**

Calcula:

* ``dti``: Debt-to-Income ratio (%)
* ``capacidad_ahorro``: Salario - Egresos
* ``capacidad_residual``: Capacidad de ahorro - Cuota
* ``ratio_patrimonio_deuda``: Patrimonio / Deuda
* ``meses_colchon``: Meses de reserva
* Y 15+ características adicionales

**Fase 6: Nivel de Riesgo**

Calcula el nivel de riesgo final:

* **Bajo**: 60% de los registros (bajo riesgo de default)
* **Medio**: 25% de los registros (riesgo moderado)
* **Alto**: 15% de los registros (alto riesgo de default)

Interpretación de Resultados
=============================

Resumen Estadístico
-------------------

Después de la generación, verás un panel con:

**Información General:**

.. code-block:: text

   ✓✓✓ GENERACIÓN COMPLETADA
   
   Total de registros: 10,000
   Total de columnas: 47
   Tiempo de generación: 8.3 segundos

**Distribución de Riesgo:**

.. code-block:: text

   Distribución de Nivel de Riesgo:
   
   Bajo:  6,074 (60.7%) ████████████████████████
   Medio: 2,943 (29.4%) ████████████
   Alto:    983 ( 9.8%) ████

.. note::
   La distribución real puede variar ligeramente del objetivo (60/25/15) debido a la aleatoriedad y las reglas de negocio.

**Estadísticas Clave:**

.. code-block:: text

   Estadísticas Principales:
   
   Edad promedio:              38.2 años
   Salario promedio:           $4,235,000 COP
   Puntaje DataCrédito:        720 puntos
   DTI promedio:               27.3%
   Capacidad residual:         $1,245,000 COP
   Valor inmueble promedio:    $185,000,000 COP

Visualizaciones
---------------

El módulo genera automáticamente:

**1. Distribución de Riesgo (Gráfico de Barras)**

Muestra la proporción de cada nivel de riesgo.

**2. Distribución de Variables Clave (Histogramas)**

* Edad
* Salario mensual
* Puntaje DataCrédito
* DTI (Debt-to-Income)
* Valor del inmueble

**3. Matriz de Correlación**

Muestra las correlaciones entre variables numéricas principales.

.. tip::
   Usa el botón de descarga en cada gráfico para guardar las visualizaciones.

Validaciones Automáticas
-------------------------

El generador valida automáticamente:

✅ **Restricción 1: Salario > Egresos**
   Todos los registros tienen capacidad de ahorro positiva.

✅ **Restricción 2: DTI ≤ 40%**
   Ningún registro excede el 40% de endeudamiento.

✅ **Restricción 3: Capacidad Residual ≥ 0**
   Todos pueden pagar la cuota y mantener gastos básicos.

✅ **Restricción 4: Edad + Plazo ≤ 80**
   El crédito termina antes de los 80 años.

✅ **Correlaciones Realistas**
   Las variables mantienen relaciones lógicas.

Si alguna validación falla, verás advertencias específicas.

Variables Generadas
===================

El dataset incluye 47 variables organizadas en categorías:

Variables Demográficas (6)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Descripción
   * - ``edad``
     - Edad del solicitante (22-65 años)
   * - ``ciudad``
     - Ciudad de residencia (15 ciudades colombianas)
   * - ``estrato_socioeconomico``
     - Estrato socioeconómico (1-6)
   * - ``nivel_educacion``
     - Nivel educativo alcanzado
   * - ``estado_civil``
     - Estado civil actual
   * - ``personas_a_cargo``
     - Número de dependientes económicos

Variables Laborales (4)
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Descripción
   * - ``tipo_empleo``
     - Formal, Informal o Independiente
   * - ``antiguedad_empleo``
     - Años en el empleo actual
   * - ``salario_mensual``
     - Ingreso mensual en COP
   * - ``egresos_mensuales``
     - Gastos mensuales totales en COP

Variables Financieras (5)
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Descripción
   * - ``numero_demandas``
     - Demandas legales por dinero (0-3)
   * - ``puntaje_datacredito``
     - Score crediticio (350-850)
   * - ``numero_propiedades``
     - Propiedades que posee (0-3)
   * - ``patrimonio_total``
     - Patrimonio neto en COP
   * - ``saldo_promedio_banco``
     - Saldo promedio últimos 6 meses

Variables del Crédito (10)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Descripción
   * - ``valor_inmueble``
     - Valor comercial de la propiedad
   * - ``anos_inmueble``
     - Antigüedad del inmueble
   * - ``porcentaje_cuota_inicial``
     - Porcentaje de cuota inicial (10-40%)
   * - ``valor_cuota_inicial``
     - Valor en COP de la cuota inicial
   * - ``monto_credito``
     - Monto solicitado del préstamo
   * - ``plazo_credito``
     - Plazo en años (10-30)
   * - ``tasa_interes_anual``
     - Tasa de interés anual (%)
   * - ``cuota_mensual``
     - Cuota mensual del crédito
   * - ``ltv``
     - Loan-to-Value ratio (%)
   * - ``dti``
     - Debt-to-Income ratio (%)

Características Derivadas (15+)
--------------------------------

Variables calculadas automáticamente:

* ``capacidad_ahorro``: Salario - Egresos
* ``capacidad_residual``: Capacidad ahorro - Cuota
* ``ratio_patrimonio_deuda``: Patrimonio / Deuda
* ``meses_colchon``: Reservas en meses
* ``score_edad``: Score basado en edad
* ``flag_sobreendeudamiento``: Indicador de sobreendeudamiento
* ``score_estabilidad_laboral``: Score de estabilidad
* ``riesgo_legal``: Score de riesgo legal
* Y más...

Variable Objetivo (1)
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Descripción
   * - ``nivel_riesgo``
     - **Bajo** / **Medio** / **Alto**

Casos de Uso Comunes
=====================

Caso 1: Generar Dataset para Entrenamiento
-------------------------------------------

**Objetivo**: Crear un dataset balanceado para entrenar modelos de ML.

**Pasos:**

1. Configura 20,000 registros
2. Usa semilla 42 para reproducibilidad
3. Genera el dataset
4. Descarga como CSV
5. Usa en módulo de modelos supervisados

**Resultado esperado:**

* 20,000 registros
* ~60% Bajo, ~25% Medio, ~15% Alto
* Listo para entrenamiento

Caso 2: Experimentar con Diferentes Distribuciones
---------------------------------------------------

**Objetivo**: Ver cómo cambian los datos con diferentes semillas.

**Pasos:**

1. Genera con semilla 42 → Observa distribución
2. Genera con semilla 123 → Compara diferencias
3. Genera con semilla 999 → Analiza variaciones

**Aprendizaje:**

* Diferentes semillas = diferentes distribuciones
* Algunas semillas pueden dar distribuciones más extremas
* Útil para validar robustez de modelos

Caso 3: Dataset Pequeño para Pruebas Rápidas
---------------------------------------------

**Objetivo**: Probar funcionalidades sin esperar.

**Pasos:**

1. Configura 1,000 registros
2. Genera en <2 segundos
3. Prueba análisis descriptivo
4. Experimenta con visualizaciones

**Ventaja:**

* Iteración rápida
* Ideal para debugging
* Bajo consumo de memoria

Caso 4: Dataset Grande para Producción
---------------------------------------

**Objetivo**: Entrenar modelos robustos para producción.

**Pasos:**

1. Configura 50,000 registros
2. Usa semilla fija para reproducibilidad
3. Genera (toma ~30-40 segundos)
4. Valida calidad exhaustivamente
5. Entrena modelos finales

**Consideraciones:**

* Mayor tiempo de generación
* Mayor consumo de memoria
* Mejor representatividad estadística

Descargar Datos
===============

Formatos Disponibles
--------------------

**1. CSV (Recomendado)**

.. code-block:: text

   [💾 Descargar CSV]

* Formato universal
* Compatible con Excel, Python, R
* Tamaño: ~5-10 MB por 10,000 registros

**2. Excel**

.. code-block:: text

   [📊 Descargar Excel]

* Formato .xlsx
* Incluye formato y estilos
* Ideal para presentaciones

**3. Parquet**

.. code-block:: text

   [⚡ Descargar Parquet]

* Formato columnar comprimido
* Más rápido para datasets grandes
* Ideal para análisis con Pandas/Spark

**4. JSON**

.. code-block:: text

   [📋 Descargar JSON]

* Formato estructurado
* Útil para APIs
* Incluye metadata

Metadata del Dataset
--------------------

Junto con los datos, se genera un archivo de metadata:

.. code-block:: json

   {
     "fecha_generacion": "2024-01-15 10:30:00",
     "numero_registros": 10000,
     "semilla_aleatoria": 42,
     "version": "1.3 - REALISTA",
     "distribucion_objetivo": "60% Bajo, 25% Medio, 15% Alto",
     "columnas": [...],
     "distribucion_riesgo": {
       "Bajo": 6074,
       "Medio": 2943,
       "Alto": 983
     },
     "estadisticas_clave": {
       "salario_promedio": 4235000,
       "edad_promedio": 38.2,
       "puntaje_datacredito_promedio": 720,
       "dti_promedio": 27.3
     }
   }

.. tip::
   Guarda siempre la metadata junto con los datos para trazabilidad.

Tips y Mejores Prácticas
=========================

Selección de Parámetros
------------------------

✅ **Haz:**

* Usa semillas fijas para experimentos reproducibles
* Comienza con 10,000 registros (balance velocidad/calidad)
* Genera múltiples datasets con diferentes semillas para validación
* Documenta la semilla usada en cada experimento

❌ **Evita:**

* Cambiar semilla constantemente sin documentar
* Generar datasets muy pequeños (<1,000) para entrenamiento
* Usar datasets muy grandes (>50,000) sin necesidad
* Modificar parámetros avanzados sin entender el impacto

Calidad de Datos
----------------

**Siempre valida:**

1. **Distribución de riesgo**: ¿Está cerca del objetivo?
2. **Rangos de variables**: ¿Son realistas?
3. **Correlaciones**: ¿Tienen sentido lógico?
4. **Valores faltantes**: No debería haber ninguno
5. **Outliers**: Deben ser razonables

**Checklist de calidad:**

.. code-block:: text

   ☑ Distribución de riesgo: 55-65% Bajo, 20-30% Medio, 10-20% Alto
   ☑ DTI promedio: 25-30%
   ☑ Capacidad residual: 100% positiva
   ☑ Puntaje DataCrédito: 650-750 promedio
   ☑ Sin valores faltantes
   ☑ Sin duplicados

Reproducibilidad
----------------

**Para garantizar reproducibilidad:**

1. **Documenta la semilla**: Anota siempre qué semilla usaste
2. **Versiona los datos**: Guarda fecha y versión del generador
3. **Exporta metadata**: Incluye toda la configuración usada
4. **Usa control de versiones**: Git para scripts de generación

**Ejemplo de documentación:**

.. code-block:: text

   Dataset: credito_hipotecario_train.csv
   Fecha: 2024-01-15
   Registros: 20,000
   Semilla: 42
   Versión generador: 1.3
   Distribución: 60.7% Bajo, 29.4% Medio, 9.8% Alto
   Propósito: Entrenamiento modelo RBM + Random Forest

Troubleshooting
===============

Problemas Comunes
-----------------

**Problema 1: Generación muy lenta**

.. code-block:: text

   Síntoma: Toma más de 1 minuto para 10,000 registros

**Causas posibles:**

* Número de registros muy alto (>50,000)
* Computadora con recursos limitados
* Otros procesos consumiendo CPU

**Soluciones:**

1. Reduce el número de registros
2. Cierra otras aplicaciones
3. Espera a que termine (no interrumpas)
4. Considera generar en lotes más pequeños

---

**Problema 2: Distribución de riesgo muy desbalanceada**

.. code-block:: text

   Síntoma: 80% Bajo, 15% Medio, 5% Alto (muy diferente del objetivo)

**Causas posibles:**

* Semilla aleatoria particular
* Número de registros muy pequeño (<5,000)

**Soluciones:**

1. Prueba con otra semilla
2. Aumenta el número de registros
3. Genera múltiples datasets y combínalos
4. Usa técnicas de balanceo posterior (SMOTE, undersampling)

---

**Problema 3: Error al descargar**

.. code-block:: text

   Error: "Failed to download file"

**Soluciones:**

1. Verifica espacio en disco
2. Intenta otro formato (CSV en lugar de Excel)
3. Reduce el tamaño del dataset
4. Recarga la página y genera nuevamente

---

**Problema 4: Valores poco realistas**

.. code-block:: text

   Síntoma: Salarios de $100M o DTI de 80%

**Causas posibles:**

* Parámetros avanzados modificados incorrectamente
* Bug en versión específica

**Soluciones:**

1. Restaura configuración por defecto
2. Verifica que no hayas modificado parámetros avanzados
3. Reporta el problema con la semilla usada
4. Usa una semilla diferente temporalmente

Errores Técnicos
----------------

**Error: "Memory Error"**

**Causa**: Dataset demasiado grande para la RAM disponible.

**Solución**:

.. code-block:: python

   # Genera en lotes más pequeños
   # Lote 1: semilla 42, 25,000 registros
   # Lote 2: semilla 43, 25,000 registros
   # Combina después

---

**Error: "Invalid seed value"**

**Causa**: Semilla fuera del rango 1-9999.

**Solución**: Usa un valor entre 1 y 9999.

---

**Error: "Generation failed"**

**Causa**: Error interno del generador.

**Solución**:

1. Recarga la página
2. Intenta con configuración por defecto
3. Reporta el error con detalles

Preguntas Frecuentes
====================

**P: ¿Cuántos registros debo generar?**

R: Depende del uso:

* Exploración: 5,000-10,000
* Entrenamiento: 20,000-30,000
* Producción: 30,000-50,000

**P: ¿Los datos son completamente aleatorios?**

R: No, mantienen correlaciones y restricciones realistas del mundo real.

**P: ¿Puedo usar estos datos en producción?**

R: Son ideales para desarrollo y pruebas. Para producción, valida con datos reales.

**P: ¿Qué semilla debo usar?**

R: Cualquiera entre 1-9999. La semilla 42 es el valor por defecto recomendado.

**P: ¿Por qué la distribución no es exactamente 60/25/15?**

R: Es un objetivo, no una garantía. La aleatoriedad y las reglas de negocio causan variaciones naturales.

**P: ¿Puedo modificar las reglas de generación?**

R: Los parámetros avanzados permiten algunos ajustes. Para cambios mayores, necesitarías modificar el código fuente.

**P: ¿Los datos incluyen valores faltantes?**

R: No, todos los registros están completos. Si necesitas simular valores faltantes, puedes eliminarlos manualmente después.

Próximos Pasos
==============

Ahora que has generado tu dataset, puedes:

1. **Explorar los datos**: :doc:`04_analisis_descriptivo`
2. **Cargar datos externos**: :doc:`03_cargar_datos`
3. **Crear características**: :doc:`05_ingenieria_caracteristicas`
4. **Entrenar RBM**: :doc:`07_rbm`

.. tip::
   Te recomendamos explorar el dataset generado con el módulo de análisis descriptivo antes de entrenar modelos.

¡Felicitaciones! Has aprendido a generar datos sintéticos realistas. 🎉