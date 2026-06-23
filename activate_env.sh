#!/bin/bash
# ============================================================================
# SCRIPT DE ACTIVACIÓN - Sistema de Riesgo Crediticio con RBM
# ============================================================================

echo "🏦 Sistema de Riesgo Crediticio con RBM"
echo "========================================"

# Verificar que existe el ambiente virtual
if [ ! -d "venv_fisica" ]; then
    echo "❌ Error: No se encontró el ambiente virtual 'venv_fisica'"
    echo "💡 Ejecuta: python3 -m venv venv_fisica"
    exit 1
fi

# Activar ambiente virtual
echo "🐍 Activando ambiente virtual..."
source venv_fisica/bin/activate

# Verificar instalación de dependencias
echo "📦 Verificando dependencias..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Instalando dependencias..."
    pip install -r requirements.txt
fi

echo "✅ Ambiente listo!"
echo ""
echo "🚀 Para ejecutar la aplicación:"
echo "   streamlit run app.py"
echo ""
echo "🌐 La aplicación se abrirá en: http://localhost:8501"
echo ""

# Mantener el shell activo con el ambiente virtual
exec bash