generar_datos
=============

.. automodule:: src.generar_datos
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Este módulo implementa un generador completo de datos sintéticos de crédito hipotecario para Colombia con distribución realista de riesgo crediticio.

**Características principales:**

* Generación de 10,000+ registros sintéticos realistas
* Distribución de riesgo: 60% Bajo, 25% Medio, 15% Alto
* Correlaciones creíbles entre variables
* Capacidad residual 100% positiva garantizada
* Validaciones automáticas de consistencia

Clases Principales
------------------

GeneradorCreditoHipotecarioRealista
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: src.generar_datos.GeneradorCreditoHipotecarioRealista
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Generador de datos sintéticos de crédito hipotecario con parámetros realistas para el contexto colombiano.
   
   **Parámetros del constructor:**
   
   :param n_registros: Número de registros a generar (default: 10000)
   :type n_registros: int
   :param semilla: Semilla aleatoria para reproducibilidad (default: 42)
   :type semilla: int
   
   **Atributos principales:**
   
   * ``n``: Número de registros
   * ``semilla_base``: Semilla aleatoria base
   * ``df``: DataFrame con datos generados
   * ``ciudades``: Distribución de ciudades colombianas
   * ``estratos_por_ciudad``: Distribución de estratos socioeconómicos
   * ``salario_base_educacion``: Rangos salariales por nivel educativo
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.generar_datos import GeneradorCreditoHipotecarioRealista
      
      # Crear generador
      generador = GeneradorCreditoHipotecarioRealista(n_registros=5000, semilla=42)
      
      # Generar datos
      df = generador.generar()
      
      # Exportar a CSV
      generador.exportar_csv("datos_credito.csv")
      generador.exportar_metadata("metadata.json")

Métodos Principales
-------------------

generar
^^^^^^^

.. automethod:: src.generar_datos.GeneradorCreditoHipotecarioRealista.generar
   :noindex:

Genera el dataset completo ejecutando todas las fases de generación.

**Returns:**
   DataFrame con todos los datos generados

**Fases de generación:**

1. Variables demográficas (edad, ciudad, estrato, educación)
2. Variables laborales (tipo empleo, antigüedad, salario)
3. Variables financieras (demandas, puntaje DataCrédito, patrimonio)
4. Variables del crédito (valor inmueble, cuota inicial, plazo)
5. Características derivadas (DTI, LTV, capacidad residual)
6. Nivel de riesgo (clasificación final)

exportar_csv
^^^^^^^^^^^^

.. automethod:: src.generar_datos.GeneradorCreditoHipotecarioRealista.exportar_csv
   :noindex:

Exporta el dataset generado a formato CSV.

**Parameters:**
   * ``nombre_archivo`` (str): Nombre del archivo de salida

exportar_metadata
^^^^^^^^^^^^^^^^^

.. automethod:: src.generar_datos.GeneradorCreditoHipotecarioRealista.exportar_metadata
   :noindex:

Exporta metadata de la generación en formato JSON.

**Parameters:**
   * ``nombre_archivo`` (str): Nombre del archivo de metadata

Funciones Auxiliares
--------------------

generar_datos_credito_realista
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.generar_datos.generar_datos_credito_realista

Función principal para generar datos de crédito hipotecario con configuración realista.

**Parameters:**
   * ``n_registros`` (int): Número de registros (default: 10000)
   * ``semilla`` (int): Semilla aleatoria (default: 42)
   * ``exportar_csv`` (bool): Si exportar a CSV (default: True)
   * ``exportar_metadata`` (bool): Si exportar metadata (default: True)

**Returns:**
   DataFrame con datos generados

**Ejemplo:**

.. code-block:: python

   from src.generar_datos import generar_datos_credito_realista
   
   # Generar 5000 registros
   df = generar_datos_credito_realista(
       n_registros=5000,
       semilla=42,
       exportar_csv=True,
       exportar_metadata=True
   )
   
   print(f"Generados {len(df)} registros")
   print(f"Columnas: {df.columns.tolist()}")

Variables Generadas
-------------------

El generador crea las siguientes variables:

**Demográficas:**
   * ``edad``: Edad del solicitante (22-65 años)
   * ``ciudad``: Ciudad de residencia
   * ``estrato_socioeconomico``: Estrato (1-6)
   * ``nivel_educacion``: Nivel educativo
   * ``estado_civil``: Estado civil
   * ``personas_a_cargo``: Número de dependientes

**Laborales:**
   * ``tipo_empleo``: Formal/Informal/Independiente
   * ``antiguedad_empleo``: Años en el empleo actual
   * ``salario_mensual``: Ingreso mensual (COP)
   * ``egresos_mensuales``: Gastos mensuales (COP)

**Financieras:**
   * ``puntaje_datacredito``: Score crediticio (150-950)
   * ``numero_demandas``: Demandas legales
   * ``patrimonio_total``: Patrimonio total (COP)
   * ``numero_propiedades``: Propiedades actuales
   * ``saldo_promedio_banco``: Saldo bancario promedio

**Del Crédito:**
   * ``valor_inmueble``: Valor del inmueble (COP)
   * ``porcentaje_cuota_inicial``: Porcentaje de cuota inicial
   * ``valor_cuota_inicial``: Valor de cuota inicial (COP)
   * ``monto_credito``: Monto del crédito (COP)
   * ``plazo_credito``: Plazo en años
   * ``tasa_interes_anual``: Tasa de interés anual (%)
   * ``cuota_mensual``: Cuota mensual (COP)

**Derivadas:**
   * ``ltv``: Loan-to-Value ratio (%)
   * ``dti``: Debt-to-Income ratio (%)
   * ``capacidad_ahorro``: Capacidad de ahorro mensual
   * ``capacidad_residual``: Capacidad residual después de cuota
   * ``nivel_riesgo``: Clasificación de riesgo (Bajo/Medio/Alto)

Validaciones Implementadas
---------------------------

El generador incluye validaciones automáticas:

1. **Salario > Egresos**: Garantizado en todos los casos
2. **DTI ≤ 40%**: Ratio de endeudamiento controlado
3. **Capacidad Residual ≥ 0**: 100% de casos con capacidad positiva
4. **Edad + Plazo ≤ 80**: Restricción de edad máxima al finalizar crédito
5. **Correlaciones realistas**: Verificación de correlaciones esperadas

Notas de Implementación
------------------------

**Mejoras realistas implementadas:**

* Distribución de riesgo ajustada a la realidad (60-25-15)
* Salarios y valores de inmuebles más conservadores
* Correlaciones suaves entre variables
* Impacto realista de demandas en puntaje DataCrédito
* DTI máximo garantizado de 35%
* Capacidad residual siempre positiva

**Configuración por defecto:**

* 10,000 registros
* Semilla aleatoria: 42
* Distribución de ciudades según población real
* Estratos según distribución urbana colombiana
* Salarios ajustados por ciudad y educación

Ver también
-----------

* :doc:`data_processor` - Procesamiento de datos generados
* :doc:`feature_engineering` - Ingeniería de características
* :doc:`supervised_models` - Modelos de clasificación