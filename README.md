# 🏦 Sistema de Riesgo Crediticio con RBM y RAG Educativo

🔗 **App desplegada:** [http://157.137.229.69:1111/](http://157.137.229.69:1111/)  
📘 **Documentación:** [http://157.137.229.69:2222/](http://157.137.229.69:2222/)
---

Sistema integral de análisis y predicción de riesgo crediticio hipotecario para Colombia, implementado con **Streamlit**, **Máquinas de Boltzmann Restringidas (RBM)** y **sistema RAG educativo**.

## 🎯 Características Principales

- 📊 **Generación de datos sintéticos** realistas para Colombia
- 📈 **Análisis exploratorio** univariado y bivariado completo
- ⚡ **Máquinas de Boltzmann Restringidas** para extracción de características
- 🤖 **9 modelos de Machine Learning** supervisados
- 🔮 **Sistema de predicción** en tiempo real
- 🎓 **Asistente RAG educativo** con papers científicos sobre RBMs
- 📚 **Documentación completa** con Sphinx

## 🐳 Instalación con Docker (recomendada)

La forma más rápida y reproducible. No necesitas instalar Python ni las dependencias
a mano: todo corre dentro de contenedores. Levanta **dos servicios**:

| Servicio | URL |
|----------|-----|
| App (Streamlit) | http://localhost:1111 |
| Documentación (Sphinx) | http://localhost:2222 |

```bash
# 1. Clonar
git clone https://github.com/fernandogomez0621/AppFIsica.git
cd AppFIsica

# 2. Configurar la(s) API key(s) de Groq
cp .env.example .env
#    edita .env y pon tu clave (ver "Configurar API Keys" más abajo)

# 3. Construir y levantar
docker compose up --build
```

> 💡 La app soporta **N API keys con failover automático**: si una falla o se queda
> sin cuota, prueba la siguiente. Más detalles en el `.env.example` y en `DOCKER.md`.

## 🚀 Instalación Manual (alternativa)

### 1. Clonar el repositorio
```bash
git clone https://github.com/fernandogomez0621/AppFIsica.git
cd AppFIsica
```

### 2. Crear ambiente virtual
```bash
python3 -m venv venv_fisica
source venv_fisica/bin/activate  # Linux/Mac
# o
venv_fisica\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar API Keys
Genera tus claves en [console.groq.com/keys](https://console.groq.com/keys) y
configúralas de **una** de estas formas:

**Opción A — variables de entorno / archivo `.env`** (recomendada, soporta varias keys):
```bash
# Una sola clave:
GROQ_API_KEY=gsk_tu_clave

# O varias con failover (lista separada por comas, en UNA línea):
GROQ_API_KEYS=gsk_clave1,gsk_clave2,gsk_clave3
```

**Opción B — `.streamlit/secrets.toml`**:
```toml
GROQ_API_KEY = "gsk_tu_clave"

[rag]
model_name = "openai/gpt-oss-120b"
```

> ⚠️ Nunca subas tus claves al repositorio. El archivo `.env` y
> `.streamlit/secrets.toml` están en `.gitignore` por seguridad. Usa
> `.env.example` como plantilla.

### 5. Ejecutar la aplicación
```bash
streamlit run app.py
```

## 📁 Estructura del Proyecto

```
AppFIsica/
│
├── app.py                          # 🚀 Aplicación principal Streamlit
├── requirements.txt                # 📦 Dependencias
├── README.md                       # 📖 Este archivo
│
├── Dockerfile                      # 🐳 Imagen de la app
├── docker-compose.yml              # 🐳 Orquestación (app + docs)
├── docker-entrypoint.sh            # 🐳 Genera secrets desde variables
├── .dockerignore
├── .env.example                    # 🔑 Plantilla de variables (sin claves)
├── DOCKER.md                       # 📘 Guía de despliegue con Docker
│
├── .streamlit/
│   └── secrets.toml               # 🔐 API keys (ignorado por git)
│
├── src/                           # 💻 Código fuente modularizado
│   ├── generar_datos.py           # 📊 Generación de datos sintéticos
│   ├── data_processor.py          # 🔧 Carga y validación de datos
│   ├── univariate_analysis.py     # 📈 Análisis univariado
│   ├── bivariate_analysis.py      # 📈 Análisis bivariado
│   ├── feature_engineering.py     # ⚙️ Ingeniería de características
│   ├── clustering.py              # 🎯 Clustering
│   ├── rbm_model.py               # ⚡ Máquina de Boltzmann Restringida
│   ├── supervised_models.py       # 🤖 Modelos supervisados
│   ├── prediction.py              # 🔮 Predicción
│   ├── retraining.py              # 🔁 Reentrenamiento
│   └── educational_rag.py         # 🎓 Sistema RAG educativo
│
├── data/                          # 💾 Datos (volumen en Docker)
├── models/                        # 🧠 Modelos entrenados (volumen en Docker)
├── articles/                      # 📚 Papers científicos (PDFs)
├── chroma_rbm_db/                 # 🗄️ Base de datos vectorial
├── docs/                          # 📘 Documentación Sphinx
└── tests/                         # 🧪 Tests unitarios
```

## 🎮 Guía de Uso

### 1. 📊 Generar Datos
- Ve a **"Generar Datos"** en el sidebar
- Configura número de registros (1K - 50K)
- Ajusta semilla aleatoria para reproducibilidad
- Genera dataset con distribución realista: 60% Bajo, 25% Medio, 15% Alto

### 2. 📁 Cargar y Validar Datos
- Sube archivos CSV, Excel o Parquet
- Ejecuta validación automática de calidad
- Aplica limpieza y preprocesamiento
- Visualiza reportes de calidad

### 3. 📈 Análisis Descriptivo
- Selecciona variables para análisis detallado
- Obtén estadísticas completas (media, mediana, percentiles, etc.)
- Visualiza distribuciones con histogramas, boxplots, Q-Q plots
- Tests de normalidad automáticos

### 4. ⚡ Entrenar RBM
- Configura arquitectura (unidades ocultas, learning rate, etc.)
- Entrena Máquina de Boltzmann Restringida
- Visualiza pesos, activaciones y curvas de aprendizaje
- Extrae características latentes para modelos supervisados

### 5. 🎓 Aprender sobre RBMs
- Chat interactivo con **Groq AI** (GPT-OSS 120B)
- Sube papers científicos en PDF
- Haz preguntas sobre Máquinas de Boltzmann
- Obtén respuestas basadas en literatura científica

## 📊 Variables del Dataset

### Variables Financieras del Crédito:
- `valor_inmueble`: Valor comercial de la propiedad (COP)
- `monto_credito`: Monto solicitado del préstamo (COP)
- `cuota_inicial`: Porcentaje de cuota inicial (%)
- `plazo_credito`: Plazo del crédito en años
- `tasa_interes`: Tasa de interés anual (%)

### Perfil Financiero del Solicitante:
- `puntaje_datacredito`: Score crediticio (150-950)
- `salario_mensual`: Ingreso mensual (COP)
- `egresos_mensuales`: Gastos mensuales totales (COP)
- `saldo_promedio_banco`: Saldo promedio últimos 6 meses (COP)
- `patrimonio_total`: Patrimonio neto (COP)
- `numero_propiedades`: Cantidad de propiedades que posee
- `numero_demandas`: Demandas legales por dinero

### Historial Laboral:
- `tipo_empleo`: Formal / Informal / Independiente
- `antiguedad_empleo`: Años en el empleo actual

### Educación y Demografía:
- `nivel_educacion`: Bachiller / Técnico / Profesional / Posgrado
- `edad`: Edad del solicitante
- `ciudad`: Ciudad de residencia (Colombia)
- `estrato_socioeconomico`: Estrato 1-6
- `estado_civil`: Soltero / Casado / Unión Libre / Divorciado
- `personas_a_cargo`: Número de dependientes

### Variable Objetivo:
- `nivel_riesgo`: **Bajo** / **Medio** / **Alto**

## 🧠 Máquinas de Boltzmann Restringidas (RBM)

### ¿Qué es una RBM?
Una **Máquina de Boltzmann Restringida** es un modelo generativo no supervisado que:
- Aprende representaciones latentes de los datos
- Usa una arquitectura de dos capas (visible + oculta)
- Se entrena con **Contrastive Divergence (CD-k)**
- Extrae características útiles para modelos supervisados

### Función de Energía:
```
E(v,h) = -∑ᵢ aᵢvᵢ - ∑ⱼ bⱼhⱼ - ∑ᵢⱼ vᵢWᵢⱼhⱼ
```

### Aplicaciones en Riesgo Crediticio:
- **Reducción de dimensionalidad** inteligente
- **Detección de patrones** ocultos en datos financieros
- **Generación de características** no lineales
- **Mejora del rendimiento** de modelos supervisados

## 🎓 Sistema RAG Educativo

### Características:
- 🤖 **Groq AI** con el modelo **GPT-OSS 120B** (con failover entre N keys)
- 📚 **Base de conocimiento** con papers científicos
- 🔍 **Búsqueda semántica** con embeddings vectoriales
- 💬 **Chat interactivo** con citación de fuentes
- 📤 **Carga automática** de PDFs

> ℹ️ El modelo `llama-3.3-70b-versatile` fue deprecado por Groq (jun-2026);
> el proyecto migró a `openai/gpt-oss-120b`. Puedes cambiarlo en la sección
> `[rag]` de `secrets.toml` o vía el entrypoint de Docker.

### Papers Incluidos:
- Hinton (2002) - Contrastive Divergence
- Hinton & Salakhutdinov (2006) - Deep Learning
- Hinton (2010) - Practical Guide to RBMs
- Salakhutdinov (2007) - Collaborative Filtering
- Y muchos más...

## 🔧 Tecnologías Utilizadas

### Frontend:
- **Streamlit** - Framework de aplicaciones web
- **Plotly** - Visualizaciones interactivas
- **Matplotlib/Seaborn** - Gráficos estadísticos

### Machine Learning:
- **Scikit-learn** - Modelos tradicionales
- **XGBoost/LightGBM** - Gradient boosting
- **Implementación custom** - RBM desde cero (NumPy)

### RAG System:
- **LangChain** - Framework RAG
- **ChromaDB** - Base de datos vectorial
- **HuggingFace** - Embeddings locales
- **Groq API** - LLM de alta velocidad (GPT-OSS 120B)
- **PyMuPDF** - Procesamiento de PDFs

### Data Processing:
- **Pandas** - Manipulación de datos
- **NumPy** - Computación numérica
- **SciPy** - Análisis estadístico

## 📈 Flujo de Trabajo

1. **📊 Generar/Cargar Datos** → Dataset de crédito hipotecario
2. **🔍 Validar y Limpiar** → Datos de alta calidad
3. **📈 Análisis Exploratorio** → Entender patrones
4. **🔧 Ingeniería de Características** → Variables derivadas
5. **⚡ Entrenar RBM** → Extraer características latentes
6. **🤖 Modelos Supervisados** → Clasificadores de riesgo
7. **🔮 Predicción** → Evaluar nuevos solicitantes
8. **🎓 Aprender** → Sistema RAG educativo

## 🎯 Casos de Uso

### Para Bancos y Entidades Financieras:
- Evaluación automática de riesgo crediticio
- Análisis de portafolio de créditos
- Detección de patrones de default
- Optimización de políticas de crédito

### Para Estudiantes de Física:
- Aprender sobre Máquinas de Boltzmann
- Entender aplicaciones de física estadística en finanzas
- Experimentar con modelos generativos
- Acceder a literatura científica especializada

### Para Data Scientists:
- Implementación de RBMs desde cero
- Comparación de modelos de ML
- Pipeline completo de ML
- Sistema RAG con papers científicos

## 🔧 Configuración Avanzada

### Hiperparámetros RBM:
```python
n_hidden = 100          # Unidades ocultas
learning_rate = 0.01    # Tasa de aprendizaje
n_epochs = 100          # Épocas de entrenamiento
batch_size = 64         # Tamaño de batch
k_cd = 1               # Pasos de Contrastive Divergence
```

### Configuración RAG:
```python
model_name = "openai/gpt-oss-120b"  # Modelo LLM (vía Groq)
chunk_size = 1500       # Tamaño de chunks de texto
chunk_overlap = 300     # Solapamiento entre chunks
top_k_results = 6       # Documentos más relevantes
temperature = 0.3       # Creatividad del LLM
```

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/

# Prueba rápida del RAG
python test_rag_simple.py
```

## 📚 Documentación

```bash
# Generar documentación con Sphinx
cd docs/
make html

# O servirla con Docker (puerto 2222)
docker compose up docs
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Geoffrey Hinton** - Por las Máquinas de Boltzmann
- **Groq** - Por la API de LLM ultra-rápida
- **Streamlit** - Por el framework de aplicaciones web
- **Comunidad científica** - Por los papers de investigación

## 📞 Soporte

- 🐛 Issues: [GitHub Issues](https://github.com/fernandogomez0621/AppFIsica/issues)

---

**Desarrollado con ❤️ para la comunidad de Física y Data Science**
