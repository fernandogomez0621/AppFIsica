# 📚 Documentación HTML - Sistema de Riesgo Crediticio con RBM

Esta carpeta contiene toda la documentación del sistema compilada en HTML.

## 🚀 Acceso Rápido

### Abrir Documentación Principal

```bash
# Opción 1: Navegador por defecto
xdg-open index.html

# Opción 2: Firefox
firefox index.html

# Opción 3: Chrome
google-chrome index.html

# Opción 4: Servidor local
python3 -m http.server 8000
# Luego visita: http://localhost:8000
```

## 📖 Páginas Principales

### 🏠 Inicio
- **Archivo**: `index.html`
- **Contenido**: Página principal con resumen del sistema

### 📖 Manual de Usuario
- **Archivo**: `user_guide.html`
- **Contenido**: Índice del manual con 11 guías detalladas

**Guías individuales** (en carpeta `user_guide/`):
1. `01_inicio.html` - Navegación e interfaz
2. `02_generar_datos.html` - Generación de datos sintéticos
3. `03_cargar_datos.html` - Carga de datos externos
4. `04_analisis_descriptivo.html` - Análisis exploratorio
5. `05_ingenieria_caracteristicas.html` - Feature engineering
6. `06_clustering.html` - Segmentación de clientes
7. `07_rbm.html` - Máquinas de Boltzmann
8. `08_modelos_supervisados.html` - Clasificadores ML
9. `09_prediccion.html` - Predicción de riesgo
10. `10_reentrenamiento.html` - Actualización de modelos
11. `11_rag_educativo.html` - Sistema RAG educativo

### ⚙️ Instalación
- **Archivo**: `installation.html`
- **Contenido**: Guía completa de instalación paso a paso

### 🔧 API del Código
- **Archivo**: `api_reference.html`
- **Contenido**: Documentación técnica de todos los módulos

**Módulos individuales** (en carpeta `api/`):
- `app.html` - Aplicación principal
- `generar_datos.html` - Generador de datos
- `data_processor.html` - Procesador de datos
- `univariate_analysis.html` - Análisis univariado
- `bivariate_analysis.html` - Análisis bivariado
- `feature_engineering.html` - Ingeniería de características
- `clustering.html` - Clustering
- `rbm_model.html` - Modelo RBM
- `supervised_models.html` - Modelos supervisados
- `prediction.html` - Predicción
- `retraining.html` - Reentrenamiento
- `educational_rag.html` - Sistema RAG

### 📋 Módulos del Sistema
- **Archivo**: `modules.html`
- **Contenido**: Descripción de arquitectura y módulos

## 📊 Información del Proyecto

- **Autores**: Andrés Fernando Gómez Rojas & Carlos Andrés Gómez Vasco
- **Año**: 2025
- **Licencia**: MIT License - Software Libre
- **Institución**: Universidad Distrital Francisco José de Caldas
- **Programa**: Pregrado en Física
- **Versión**: 1.0.0

## 🔗 Enlaces

- **Repositorio**: https://github.com/fernandogomez0621/AppFIsica
- **Issues**: https://github.com/fernandogomez0621/AppFIsica/issues
- **Email**: fernandogomez0621@gmail.com

## 📁 Estructura de Archivos

```
docs/build/html/
├── index.html              # Página principal
├── user_guide.html         # Índice manual de usuario
├── installation.html       # Guía de instalación
├── api_reference.html      # API del código
├── modules.html            # Módulos del sistema
├── genindex.html           # Índice general
├── search.html             # Búsqueda
│
├── user_guide/             # 11 guías del manual
│   ├── 01_inicio.html
│   ├── 02_generar_datos.html
│   ├── 03_cargar_datos.html
│   ├── 04_analisis_descriptivo.html
│   ├── 05_ingenieria_caracteristicas.html
│   ├── 06_clustering.html
│   ├── 07_rbm.html
│   ├── 08_modelos_supervisados.html
│   ├── 09_prediccion.html
│   ├── 10_reentrenamiento.html
│   └── 11_rag_educativo.html
│
├── api/                    # 12 módulos documentados
│   ├── app.html
│   ├── generar_datos.html
│   └── ... (10 más)
│
├── _static/                # CSS, JS, imágenes
├── _sources/               # Archivos fuente RST
└── _modules/               # Código fuente resaltado
```

## 📊 Estadísticas

- **Total páginas HTML**: 42
- **Tamaño total**: 15 MB
- **Guías de usuario**: 11 (5,506 líneas RST)
- **Módulos API**: 12
- **Idioma**: Español

## 💡 Consejos de Uso

### Navegación
- Usa el menú lateral para navegar entre secciones
- La búsqueda está disponible en la esquina superior derecha
- Cada página tiene enlaces de navegación (anterior/siguiente)

### Compartir
Para compartir la documentación:

```bash
# Comprimir carpeta
tar -czf documentacion_sistema_rbm.tar.gz docs/build/html/

# Enviar archivo comprimido
# El receptor solo necesita descomprimir y abrir index.html
```

### Servidor Local
Para mejor experiencia, usa un servidor local:

```bash
cd docs/build/html
python3 -m http.server 8000
```

Luego visita: http://localhost:8000

## 🎯 Próximos Pasos

1. **Explora la documentación**: Abre `index.html`
2. **Lee el manual de usuario**: Ve a `user_guide.html`
3. **Instala el sistema**: Sigue `installation.html`
4. **Consulta la API**: Revisa `api_reference.html`

---

**Generado con Sphinx 8.2.3** | **© 2025 Andrés Fernando Gómez Rojas & Carlos Andrés Gómez Vasco** | **Software Libre - MIT License**