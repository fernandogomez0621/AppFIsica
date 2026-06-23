Sistema de Riesgo Crediticio con RBM
====================================

Bienvenido a la documentación del **Sistema de Riesgo Crediticio con Máquinas de Boltzmann Restringidas (RBM)**.

Este sistema integral permite el análisis y predicción de riesgo crediticio hipotecario para Colombia, 
utilizando técnicas avanzadas de Machine Learning y un sistema RAG educativo.

.. toctree::
   :maxdepth: 2
   :caption: Contenidos:

   installation
   user_guide
   modules
   api_reference

Características Principales
==========================

🎯 **Objetivo del Proyecto**
---------------------------

Crear un sistema integral que permita:

* 📊 **Generar/cargar datos** de solicitudes de crédito hipotecario
* 📈 **Realizar análisis exploratorio** avanzado  
* ⚙️ **Aplicar ingeniería de características** automática
* 🧠 **Entrenar modelos predictivos** con RBM + clasificadores
* 🔮 **Predecir riesgo crediticio** en nuevos solicitantes
* 🎓 **Aprender sobre Máquinas de Boltzmann** mediante un asistente RAG

📊 **Variables del Sistema**
---------------------------

Variables Financieras del Crédito:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``valor_inmueble``: Valor comercial de la propiedad (COP)
* ``monto_credito``: Monto solicitado del préstamo (COP)  
* ``cuota_inicial``: Porcentaje de cuota inicial (%)
* ``plazo_credito``: Plazo del crédito en años
* ``tasa_interes``: Tasa de interés anual (%)

Perfil Financiero del Solicitante:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``puntaje_datacredito``: Score crediticio (150-950)
* ``salario_mensual``: Ingreso mensual (COP)
* ``egresos_mensuales``: Gastos mensuales totales (COP)
* ``saldo_promedio_banco``: Saldo promedio últimos 6 meses (COP)
* ``patrimonio_total``: Patrimonio neto (COP)
* ``numero_propiedades``: Cantidad de propiedades que posee
* ``numero_demandas``: Demandas legales por dinero

Variable Objetivo:
~~~~~~~~~~~~~~~~~

* ``nivel_riesgo``: **Bajo** / **Medio** / **Alto**

⚡ **Máquinas de Boltzmann Restringidas**
----------------------------------------

¿Qué es una RBM?
~~~~~~~~~~~~~~~

Una **Máquina de Boltzmann Restringida** es un modelo generativo no supervisado que:

* Aprende representaciones latentes de los datos
* Usa una arquitectura de dos capas (visible + oculta)
* Se entrena con **Contrastive Divergence (CD-k)**
* Extrae características útiles para modelos supervisados

Función de Energía:
~~~~~~~~~~~~~~~~~~

.. math::

   E(v,h) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i W_{ij} h_j

Donde:

* :math:`v_i` son las unidades visibles
* :math:`h_j` son las unidades ocultas  
* :math:`W_{ij}` son los pesos de conexión
* :math:`a_i, b_j` son los sesgos

🎓 **Sistema RAG Educativo**
---------------------------

Características:
~~~~~~~~~~~~~~~

* 🤖 **Groq AI** con Llama 3.3 70B parámetros
* 📚 **Base de conocimiento** con papers científicos
* 🔍 **Búsqueda semántica** con embeddings vectoriales
* 💬 **Chat interactivo** con citación de fuentes
* 📤 **Carga automática** de PDFs

Inicio Rápido
=============

Instalación
-----------

.. code-block:: bash

   # Clonar repositorio
   git clone <tu-repositorio>
   cd AppFIsica
   
   # Crear ambiente virtual
   python3 -m venv venv_fisica
   source venv_fisica/bin/activate
   
   # Instalar dependencias
   pip install -r requirements.txt

Ejecución
---------

.. code-block:: bash

   # Activar ambiente virtual
   source venv_fisica/bin/activate
   
   # Ejecutar aplicación
   streamlit run app.py

La aplicación se abrirá en: http://localhost:8501

Flujo de Trabajo
===============

1. **📊 Generar/Cargar Datos** → Dataset de crédito hipotecario
2. **🔍 Validar y Limpiar** → Datos de alta calidad  
3. **📈 Análisis Exploratorio** → Entender patrones
4. **🔧 Ingeniería de Características** → Variables derivadas
5. **⚡ Entrenar RBM** → Extraer características latentes
6. **🤖 Modelos Supervisados** → Clasificadores de riesgo
7. **🔮 Predicción** → Evaluar nuevos solicitantes
8. **🎓 Aprender** → Sistema RAG educativo

Índices y Tablas
================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`