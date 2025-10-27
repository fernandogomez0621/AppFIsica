# 📚 Papers Científicos - Máquinas de Boltzmann

Esta carpeta contiene los papers científicos en formato PDF que alimentan el sistema RAG educativo.

## 🔄 Cómo agregar papers

### Opción 1: Interfaz de Streamlit
1. Ejecuta la aplicación: `streamlit run app.py`
2. Ve a la sección "🎓 Aprende sobre RBMs"
3. Usa la pestaña "📤 Subir Papers"
4. Arrastra y suelta los PDFs o selecciónalos

### Opción 2: Script automático
```bash
# Ejecutar el script de descarga automática
python libros.py
```

### Opción 3: Manual
1. Descarga papers de fuentes confiables:
   - [arXiv.org](https://arxiv.org/search/?query=restricted+boltzmann+machine)
   - [Google Scholar](https://scholar.google.com/)
   - [Semantic Scholar](https://www.semanticscholar.org/)

2. Guarda los PDFs en esta carpeta

3. Re-indexa la base de conocimiento desde la aplicación

## 📋 Papers recomendados

### Fundamentos
- **Hinton (2002)** - Training Products of Experts by Minimizing Contrastive Divergence
- **Hinton & Salakhutdinov (2006)** - Reducing the Dimensionality of Data with Neural Networks
- **Hinton (2010)** - A Practical Guide to Training Restricted Boltzmann Machines

### Algoritmos
- **Tieleman (2008)** - Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient
- **Salakhutdinov (2009)** - Deep Boltzmann Machines

### Aplicaciones
- **Salakhutdinov (2007)** - Restricted Boltzmann Machines for Collaborative Filtering
- **Larochelle (2008)** - Classification using Discriminative Restricted Boltzmann Machines

## ⚠️ Notas importantes

- Solo archivos PDF son soportados
- Los papers deben estar en inglés para mejor procesamiento
- El sistema procesa automáticamente múltiples columnas
- Se extraen metadatos como título, autor, etc.
- Los papers se dividen en chunks para búsqueda semántica

## 🔧 Troubleshooting

Si tienes problemas:
1. Verifica que los PDFs no estén corruptos
2. Asegúrate de que no estén protegidos por contraseña
3. Re-indexa la base de datos desde la aplicación
4. Revisa los logs en la consola de Streamlit