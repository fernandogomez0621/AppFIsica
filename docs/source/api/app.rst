Aplicación Principal (app.py)
==============================

Módulo principal de la aplicación Streamlit que coordina todos los componentes del sistema.

.. automodule:: app
   :members:
   :undoc-members:
   :show-inheritance:

Funciones Principales
---------------------

.. autofunction:: app.main

.. autofunction:: app.render_home

.. autofunction:: app.render_data_generator

.. autofunction:: app.show_app_info

Configuración de la Página
---------------------------

La aplicación utiliza la siguiente configuración de Streamlit:

* **Título**: Sistema de Riesgo Crediticio con RBM
* **Icono**: 🏦
* **Layout**: Wide (ancho completo)
* **Sidebar**: Expandido por defecto

Navegación
----------

El sistema incluye los siguientes módulos accesibles desde el sidebar:

1. 🏠 Inicio
2. 📊 Generar Datos
3. 📁 Cargar Datos
4. 📈 Análisis Descriptivo
5. 🔧 Ingeniería de Características
6. 🎯 Clustering
7. ⚡ Máquina de Boltzmann (RBM)
8. 🤖 Modelos Supervisados
9. 📊 Comparación de Modelos
10. 🔮 Predicción
11. 🔄 Re-entrenamiento
12. 🎓 Aprende sobre RBMs
13. 📚 Documentación