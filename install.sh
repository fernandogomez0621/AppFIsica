#!/bin/bash

# ============================================================================
# Script de Instalación Automática - Sistema de Riesgo Crediticio con RBM
# ============================================================================
# Autor: Sistema de Riesgo Crediticio
# Versión: 1.0
# Fecha: 2024
# Repositorio: https://github.com/fernandogomez0621/AppFIsica.git
# ============================================================================

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                                                                      ║${NC}"
    echo -e "${BLUE}║   INSTALACIÓN - Sistema de Riesgo Crediticio con RBM                ║${NC}"
    echo -e "${BLUE}║                                                                      ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# VERIFICACIONES PREVIAS
# ============================================================================

print_header

echo "🔍 Verificando requisitos del sistema..."
echo ""

# Verificar Python
print_info "Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_success "Python $PYTHON_VERSION encontrado"
    else
        print_error "Python 3.8+ requerido. Versión actual: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 no encontrado. Por favor instala Python 3.8 o superior."
    exit 1
fi

# Verificar pip
print_info "Verificando pip..."
if command -v pip3 &> /dev/null; then
    PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
    print_success "pip $PIP_VERSION encontrado"
else
    print_error "pip no encontrado. Instalando..."
    python3 -m ensurepip --upgrade
fi

# Verificar git
print_info "Verificando git..."
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    print_success "git $GIT_VERSION encontrado"
else
    print_warning "git no encontrado. Necesitarás clonar el repositorio manualmente."
fi

# Verificar espacio en disco
print_info "Verificando espacio en disco..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -ge 3 ]; then
    print_success "Espacio disponible: ${AVAILABLE_SPACE}GB"
else
    print_warning "Espacio limitado: ${AVAILABLE_SPACE}GB (se recomiendan 3GB+)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# INSTALACIÓN
# ============================================================================

echo "📦 Iniciando instalación..."
echo ""

# Paso 1: Crear ambiente virtual
print_info "Paso 1/5: Creando ambiente virtual..."
if [ -d "venv_fisica" ]; then
    print_warning "Ambiente virtual ya existe. Eliminando..."
    rm -rf venv_fisica
fi

python3 -m venv venv_fisica
print_success "Ambiente virtual creado: venv_fisica/"

# Paso 2: Activar ambiente virtual
print_info "Paso 2/5: Activando ambiente virtual..."
source venv_fisica/bin/activate
print_success "Ambiente virtual activado"

# Paso 3: Actualizar pip
print_info "Paso 3/5: Actualizando pip..."
pip install --upgrade pip setuptools wheel --quiet
PIP_NEW_VERSION=$(pip --version | cut -d' ' -f2)
print_success "pip actualizado a versión $PIP_NEW_VERSION"

# Paso 4: Instalar dependencias
print_info "Paso 4/5: Instalando dependencias (esto puede tomar 10-15 minutos)..."
echo ""

if [ -f "requirements.txt" ]; then
    # Instalar con barra de progreso
    pip install -r requirements.txt --progress-bar on
    print_success "Todas las dependencias instaladas correctamente"
else
    print_error "Archivo requirements.txt no encontrado"
    exit 1
fi

# Paso 5: Crear directorios necesarios
print_info "Paso 5/5: Creando estructura de directorios..."
mkdir -p data/raw data/processed data/synthetic
mkdir -p models/rbm models/supervised models/versions
mkdir -p articles
mkdir -p chroma_rbm_db
mkdir -p tests
print_success "Directorios creados"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# VERIFICACIÓN DE INSTALACIÓN
# ============================================================================

echo "🔍 Verificando instalación..."
echo ""

# Verificar imports críticos
print_info "Verificando librerías críticas..."

python3 << EOF
import sys
try:
    import streamlit
    print("✓ Streamlit:", streamlit.__version__)
    import pandas
    print("✓ Pandas:", pandas.__version__)
    import numpy
    print("✓ NumPy:", numpy.__version__)
    import sklearn
    print("✓ Scikit-learn:", sklearn.__version__)
    import tensorflow
    print("✓ TensorFlow:", tensorflow.__version__)
    import plotly
    print("✓ Plotly:", plotly.__version__)
    import langchain
    print("✓ LangChain:", langchain.__version__)
    import chromadb
    print("✓ ChromaDB:", chromadb.__version__)
    print("\n✅ Todas las librerías críticas instaladas correctamente")
    sys.exit(0)
except ImportError as e:
    print(f"\n❌ Error importando librería: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Verificación de librerías completada"
else
    print_error "Algunas librerías no se instalaron correctamente"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

echo "⚙️  Configuración..."
echo ""

# Verificar archivo de secrets
if [ ! -f ".streamlit/secrets.toml" ]; then
    print_warning "Archivo .streamlit/secrets.toml no encontrado"
    print_info "Creando archivo de configuración de ejemplo..."
    
    mkdir -p .streamlit
    cat > .streamlit/secrets.toml << 'SECRETS_EOF'
# ============================================================================
# CONFIGURACIÓN DE API KEYS
# ============================================================================

# API Key de Groq (GRATIS en https://console.groq.com/keys)
GROQ_API_KEY = "tu-api-key-aqui"

# Configuración opcional
[rbm]
default_n_hidden = 100
default_learning_rate = 0.01
default_n_epochs = 100

[rag]
chunk_size = 1500
chunk_overlap = 300
top_k_results = 6
temperature = 0.3
SECRETS_EOF
    
    print_success "Archivo .streamlit/secrets.toml creado"
    print_warning "⚠️  IMPORTANTE: Edita .streamlit/secrets.toml y agrega tu GROQ_API_KEY"
    print_info "   Obtén tu API key gratis en: https://console.groq.com/keys"
else
    print_success "Archivo de configuración encontrado"
fi

# Verificar archivo de configuración de Streamlit
if [ ! -f ".streamlit/config.toml" ]; then
    print_info "Creando configuración de Streamlit..."
    cat > .streamlit/config.toml << 'CONFIG_EOF'
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
CONFIG_EOF
    print_success "Configuración de Streamlit creada"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# RESUMEN DE INSTALACIÓN
# ============================================================================

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                      ║${NC}"
echo -e "${GREEN}║   ✅ INSTALACIÓN COMPLETADA EXITOSAMENTE                            ║${NC}"
echo -e "${GREEN}║                                                                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "📊 Resumen de Instalación:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Python:              $PYTHON_VERSION"
echo "  pip:                 $PIP_NEW_VERSION"
echo "  Ambiente virtual:    venv_fisica/"
echo "  Dependencias:        ✅ Instaladas"
echo "  Configuración:       ✅ Creada"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📋 Próximos Pasos:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Configura tu API Key de Groq:"
echo "     ${YELLOW}nano .streamlit/secrets.toml${NC}"
echo "     Obtén tu key gratis en: https://console.groq.com/keys"
echo ""
echo "  2. Ejecuta la aplicación:"
echo "     ${GREEN}./activate_env.sh${NC}"
echo "     o manualmente:"
echo "     ${GREEN}source venv_fisica/bin/activate${NC}"
echo "     ${GREEN}streamlit run app.py${NC}"
echo ""
echo "  3. Abre en tu navegador:"
echo "     ${BLUE}http://localhost:8501${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📚 Documentación:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Manual de Usuario:   docs/build/html/user_guide.html"
echo "  API Reference:       docs/build/html/api_reference.html"
echo "  Instalación:         docs/build/html/installation.html"
echo ""
echo "  Abrir documentación:"
echo "  ${GREEN}firefox docs/build/html/index.html${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🎉 ¡Instalación completada! El sistema está listo para usar."
echo ""
echo "⚠️  RECUERDA: Configura tu GROQ_API_KEY antes de usar el sistema RAG educativo"
echo ""