Instalación
===========

Esta guía te ayudará a instalar y configurar el Sistema de Riesgo Crediticio con RBM.

Requisitos del Sistema
---------------------

Requisitos Mínimos
~~~~~~~~~~~~~~~~~~

* **Python**: 3.12.3 (recomendado) o 3.8+ (mínimo)
* **RAM**: 4 GB mínimo, 8 GB recomendado
* **Espacio en disco**: 3 GB para dependencias y modelos
* **Sistema operativo**: Linux, macOS, o Windows 10/11

Dependencias Principales
~~~~~~~~~~~~~~~~~~~~~~~

El sistema requiere las siguientes librerías (ver ``requirements.txt`` completo):

**Framework Web:**

* ``streamlit>=1.28.0`` - Framework de aplicaciones web

**Procesamiento de Datos:**

* ``pandas>=2.0.0`` - Manipulación de datos
* ``numpy>=1.24.0`` - Computación numérica
* ``scipy>=1.11.0`` - Análisis estadístico
* ``statsmodels>=0.14.0`` - Modelos estadísticos

**Visualización:**

* ``plotly>=5.15.0`` - Gráficos interactivos
* ``matplotlib>=3.7.0`` - Gráficos estáticos
* ``seaborn>=0.12.0`` - Visualizaciones estadísticas
* ``altair>=5.0.0`` - Gráficos declarativos

**Machine Learning:**

* ``scikit-learn>=1.3.0`` - Modelos tradicionales
* ``xgboost>=1.7.0`` - Gradient boosting
* ``lightgbm>=4.0.0`` - Gradient boosting eficiente
* ``tensorflow>=2.13.0`` - Deep learning y RBM

**Sistema RAG:**

* ``langchain>=0.0.350`` - Framework RAG
* ``langchain-community>=0.0.10`` - Integraciones
* ``langchain-groq>=0.0.3`` - Cliente Groq
* ``chromadb>=0.4.15`` - Base de datos vectorial
* ``sentence-transformers>=2.2.2`` - Embeddings
* ``transformers>=4.35.0`` - Modelos de lenguaje
* ``torch>=2.1.0`` - PyTorch
* ``groq>=0.4.0`` - API Groq

**Procesamiento de PDFs:**

* ``PyMuPDF>=1.23.0`` - Extracción de texto de PDFs
* ``fitz>=0.0.1.dev2`` - Procesamiento de documentos

**Formatos de Datos:**

* ``openpyxl>=3.1.0`` - Excel
* ``xlsxwriter>=3.1.0`` - Escritura Excel
* ``pyarrow>=12.0.0`` - Parquet

**Utilidades:**

* ``python-dotenv>=1.0.0`` - Variables de entorno
* ``pydantic>=2.4.0`` - Validación de datos
* ``tqdm>=4.66.0`` - Barras de progreso
* ``joblib>=1.3.0`` - Serialización
* ``requests>=2.31.0`` - HTTP requests
* ``arxiv>=1.4.8`` - Descarga de papers

**Documentación:**

* ``sphinx>=7.1.0`` - Generador de documentación
* ``sphinx-rtd-theme>=1.3.0`` - Tema ReadTheDocs

**Testing:**

* ``pytest>=7.4.0`` - Framework de testing
* ``pytest-cov>=4.1.0`` - Cobertura de código

**Desarrollo:**

* ``black>=23.9.0`` - Formateador de código
* ``flake8>=6.1.0`` - Linter
* ``isort>=5.12.0`` - Ordenador de imports

.. note::
   Total de dependencias: 72 paquetes principales + sus dependencias transitivas (~200 paquetes totales)

Instalación Paso a Paso
-----------------------

1. Clonar el Repositorio
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/fernandogomez0621/AppFIsica.git
   cd AppFIsica

2. Crear Ambiente Virtual
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Linux/macOS
   python3 -m venv venv_fisica
   source venv_fisica/bin/activate
   
   # Windows
   python -m venv venv_fisica
   venv_fisica\Scripts\activate

3. Instalar Dependencias
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt

.. note::
   La instalación puede tomar 10-15 minutos debido a las dependencias de deep learning.

4. Configurar API Keys
~~~~~~~~~~~~~~~~~~~~~

Edita el archivo ``.streamlit/secrets.toml``:

.. code-block:: toml

   # API Key de Groq (GRATIS en console.groq.com)
   GROQ_API_KEY = "tu-api-key-aqui"

5. Ejecutar la Aplicación
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   streamlit run app.py

La aplicación se abrirá automáticamente en: http://localhost:8501

Instalación Automática (Linux/macOS)
------------------------------------

Script de Instalación Completa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usa el script de instalación automática:

.. code-block:: bash

   chmod +x install.sh
   ./install.sh

Este script:

* ✅ Verifica Python 3.8+
* ✅ Crea ambiente virtual automáticamente
* ✅ Instala todas las dependencias
* ✅ Configura directorios necesarios
* ✅ Verifica la instalación
* ✅ Proporciona instrucciones de uso

Script de Activación Rápida
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Para sesiones posteriores, usa:

.. code-block:: bash

   chmod +x activate_env.sh
   ./activate_env.sh

Este script:

* Activa el ambiente virtual
* Verifica dependencias
* Ejecuta la aplicación Streamlit

Verificación de Instalación
---------------------------

Verificar Dependencias
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Ejecutar en Python
   import streamlit
   import pandas
   import numpy
   import plotly
   import sklearn
   import tensorflow
   print("✅ Todas las dependencias instaladas correctamente")

Verificar Aplicación
~~~~~~~~~~~~~~~~~~~

1. Ejecuta ``streamlit run app.py``
2. Ve a http://localhost:8501
3. Verifica que aparezca el dashboard principal
4. Prueba generar datos sintéticos

Solución de Problemas
--------------------

Errores Comunes
~~~~~~~~~~~~~~

**Error: ModuleNotFoundError**

.. code-block:: bash

   # Verificar que el ambiente virtual esté activo
   which python
   
   # Reinstalar dependencias
   pip install -r requirements.txt

**Error: GROQ_API_KEY not found**

* Verifica que ``.streamlit/secrets.toml`` existe
* Obtén tu API key en: https://console.groq.com/keys
* Asegúrate de que el formato sea correcto

**Error: inotify watch limit reached**

.. code-block:: bash

   # Aumentar límite de inotify (Linux)
   echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p

**Problemas de Memoria**

* Reduce ``batch_size`` en configuración RBM
* Usa menos datos para entrenamiento inicial
* Cierra otras aplicaciones

Configuración Avanzada
---------------------

Variables de Entorno
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Configurar límites de memoria para TensorFlow
   export TF_CPP_MIN_LOG_LEVEL=2
   export CUDA_VISIBLE_DEVICES=""  # Usar solo CPU
   
   # Configurar Streamlit
   export STREAMLIT_SERVER_PORT=8501
   export STREAMLIT_SERVER_HEADLESS=true

Configuración de Streamlit
~~~~~~~~~~~~~~~~~~~~~~~~~

Edita ``.streamlit/config.toml`` para personalizar:

.. code-block:: toml

   [theme]
   primaryColor = "#FF6B6B"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"

Configuración de Modelos
~~~~~~~~~~~~~~~~~~~~~~~

Ajusta hiperparámetros en ``.streamlit/secrets.toml``:

.. code-block:: toml

   [rbm]
   default_n_hidden = 100
   default_learning_rate = 0.01
   default_n_epochs = 100

Actualización
------------

Para actualizar a una nueva versión:

.. code-block:: bash

   git pull origin main
   pip install -r requirements.txt --upgrade
   streamlit run app.py

Desinstalación
-------------

Para remover completamente:

.. code-block:: bash

   # Desactivar ambiente virtual
   deactivate
   
   # Eliminar directorio del proyecto
   rm -rf AppFIsica/

Soporte
-------

* 📧 **Email**: fernandogomez0621@gmail.com
* 🐛 **Issues**: https://github.com/fernandogomez0621/AppFIsica/issues
* 📖 **Documentación**: Esta documentación
* 💬 **Chat**: Sistema RAG integrado para preguntas sobre RBMs

Próximos Pasos
-------------

Después de la instalación:

1. **Genera datos**: Ve al módulo "Generar Datos"
2. **Explora**: Usa "Análisis Descriptivo"  
3. **Entrena RBM**: Experimenta con "Máquina de Boltzmann"
4. **Aprende**: Usa el sistema RAG educativo