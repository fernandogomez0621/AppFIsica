# 🚀 Instrucciones de Uso Rápido

## ⚡ Inicio Rápido

### 1. Activar ambiente virtual y ejecutar
```bash
# Opción 1: Script automático
./activate_env.sh

# Opción 2: Manual
source venv_fisica/bin/activate
streamlit run app.py
```

### 2. Abrir en navegador
La aplicación se abrirá automáticamente en: **http://localhost:8501**

## 📊 Flujo de Trabajo Recomendado

### Paso 1: Generar Datos (✅ YA COMPLETADO)
- Ve a **"📊 Generar Datos"**
- Los datos ya están generados: **10,000 registros**
- Distribución: 60.7% Bajo, 29.4% Medio, 9.8% Alto

### Paso 2: Explorar Datos
- Ve a **"📈 Análisis Descriptivo"**
- Selecciona variables como:
  - `salario_mensual`
  - `puntaje_datacredito`
  - `dti` (Debt-to-Income ratio)
  - `nivel_riesgo`

### Paso 3: Entrenar RBM
- Ve a **"⚡ Máquina de Boltzmann (RBM)"**
- Configura hiperparámetros:
  - Unidades ocultas: 100
  - Learning rate: 0.01
  - Épocas: 100
- Haz clic en **"🎯 Entrenar RBM"**

### Paso 4: Aprender sobre RBMs
- Ve a **"🎓 Aprende sobre RBMs"**
- Sube papers científicos o usa el script automático:
  ```bash
  cd src && python libros.py
  ```
- Haz preguntas como:
  - "¿Qué es una Máquina de Boltzmann Restringida?"
  - "¿Cómo funciona Contrastive Divergence?"

## 🔧 Módulos Disponibles

| Módulo | Estado | Descripción |
|--------|--------|-------------|
| 🏠 Inicio | ✅ Completo | Dashboard principal |
| 📊 Generar Datos | ✅ Completo | Generador sintético realista |
| 📁 Cargar Datos | ✅ Completo | Carga y validación |
| 📈 Análisis Descriptivo | ✅ Completo | Estadísticas univariadas |
| ⚡ RBM | ✅ Completo | Máquina de Boltzmann |
| 🎓 RAG Educativo | ✅ Completo | Chat con papers científicos |
| 🔧 Ingeniería | 🚧 Placeholder | Características derivadas |
| 🎯 Clustering | 🚧 Placeholder | Segmentación |
| 🤖 Supervisados | 🚧 Placeholder | Modelos ML |
| 🔮 Predicción | 🚧 Placeholder | Sistema predicción |

## 📚 Datos Generados

### Estadísticas del Dataset:
- **Total registros**: 10,000
- **Variables**: 47
- **Distribución de riesgo**:
  - Bajo: 6,074 (60.7%)
  - Medio: 2,943 (29.4%)
  - Alto: 983 (9.8%)

### Variables Principales:
- `edad`: 18-80 años
- `salario_mensual`: $1M - $50M COP
- `puntaje_datacredito`: 150-950
- `valor_inmueble`: $20M - $2B COP
- `dti`: 0-60% (Debt-to-Income)
- `ltv`: 0-100% (Loan-to-Value)

## 🎯 Casos de Uso Inmediatos

### Para Estudiantes de Física:
1. **Explorar RBMs**: Ve al módulo educativo y pregunta sobre física estadística
2. **Experimentar**: Entrena RBMs con diferentes hiperparámetros
3. **Visualizar**: Observa matrices de pesos y activaciones

### Para Analistas de Datos:
1. **Análisis exploratorio**: Usa el módulo de análisis descriptivo
2. **Calidad de datos**: Valida y limpia datasets
3. **Feature engineering**: Extrae características con RBM

### Para Desarrolladores:
1. **Código fuente**: Revisa implementación en `src/`
2. **Extensibilidad**: Agrega nuevos módulos
3. **Testing**: Ejecuta `pytest tests/`

## ⚠️ Notas Importantes

### API Keys:
- **Groq API**: Ya configurada en `.streamlit/secrets.toml`
- **Límites**: 6,000 tokens/minuto (plan gratuito)

### Rendimiento:
- **RBM**: Puede tomar 2-5 minutos entrenar
- **RAG**: Primera consulta más lenta (carga embeddings)
- **Datos**: 10K registros cargan instantáneamente

### Troubleshooting:
- **Error de imports**: Verifica que estés en el ambiente virtual
- **Memoria**: Reduce batch_size si hay problemas
- **Papers**: Sube PDFs en formato estándar

## 🎉 ¡Listo para usar!

La aplicación está completamente funcional. El generador de datos ya creó un dataset realista de 10,000 registros de crédito hipotecario colombiano.

**¡Explora los módulos y experimenta con las Máquinas de Boltzmann!** 🚀