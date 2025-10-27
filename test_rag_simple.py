#!/usr/bin/env python3
"""
Test simple del sistema RAG con un solo PDF
"""

import os
import sys

def test_basic_imports():
    """Prueba imports básicos"""
    print("🔍 Probando imports básicos...")
    
    try:
        import fitz
        print("✅ PyMuPDF (fitz) disponible")
    except ImportError as e:
        print(f"❌ PyMuPDF: {e}")
        return False
    
    try:
        import groq
        print("✅ Groq disponible")
    except ImportError as e:
        print(f"❌ Groq: {e}")
        return False
    
    return True

def test_pdf_processing():
    """Prueba procesamiento de un PDF"""
    print("\n📚 Probando procesamiento de PDF...")
    
    # Buscar un PDF en articles/
    articles_dir = "./articles/"
    if not os.path.exists(articles_dir):
        print(f"❌ Directorio {articles_dir} no existe")
        return False
    
    pdf_files = [f for f in os.listdir(articles_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("❌ No hay PDFs en articles/")
        return False
    
    # Tomar el primer PDF
    test_pdf = os.path.join(articles_dir, pdf_files[0])
    print(f"📄 Probando con: {pdf_files[0]}")
    
    try:
        import fitz
        
        doc = fitz.open(test_pdf)
        print(f"✅ PDF abierto: {len(doc)} páginas")
        
        # Extraer texto de la primera página
        if len(doc) > 0:
            page = doc[0]
            text = page.get_text("text", sort=True)
            print(f"✅ Texto extraído: {len(text)} caracteres")
            
            if len(text) > 100:
                print(f"📝 Muestra: {text[:200]}...")
            else:
                print("⚠️ Poco texto extraído")
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"❌ Error procesando PDF: {e}")
        return False

def test_groq_api():
    """Prueba conexión con Groq API"""
    print("\n🤖 Probando Groq API...")
    
    try:
        # Leer API key
        import toml
        
        secrets_path = ".streamlit/secrets.toml"
        if not os.path.exists(secrets_path):
            print(f"❌ Archivo {secrets_path} no existe")
            return False
        
        with open(secrets_path, 'r') as f:
            secrets = toml.load(f)
        
        api_key = secrets.get('GROQ_API_KEY')
        if not api_key:
            print("❌ GROQ_API_KEY no encontrada en secrets.toml")
            return False
        
        print("✅ API Key encontrada")
        
        # Probar conexión básica
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # Test simple
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "¿Qué es una RBM? Responde en una línea."}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=100,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        print(f"✅ Groq respuesta: {answer[:100]}...")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error Groq API: {e}")
        return False

def test_simple_rag():
    """Prueba RAG simplificado sin LangChain"""
    print("\n🔍 Probando RAG simplificado...")
    
    try:
        # 1. Procesar un PDF
        articles_dir = "./articles/"
        pdf_files = [f for f in os.listdir(articles_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("❌ No hay PDFs")
            return False
        
        test_pdf = os.path.join(articles_dir, pdf_files[0])
        
        import fitz
        doc = fitz.open(test_pdf)
        
        # Extraer todo el texto
        full_text = ""
        for page in doc:
            text = page.get_text("text", sort=True)
            full_text += text + "\n"
        
        doc.close()
        
        print(f"✅ Texto extraído: {len(full_text)} caracteres")
        
        # 2. Hacer pregunta simple a Groq con contexto
        import toml
        from groq import Groq
        
        with open(".streamlit/secrets.toml", 'r') as f:
            secrets = toml.load(f)
        
        client = Groq(api_key=secrets['GROQ_API_KEY'])
        
        # Tomar solo una parte del texto para no exceder límites
        context = full_text[:3000]  # Primeros 3000 caracteres
        
        prompt = f"""Basándote en este contexto de un paper científico:

{context}

Pregunta: ¿Qué es una Máquina de Boltzmann Restringida?

Responde de forma educativa para estudiantes de física."""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        print(f"✅ RAG funcionando!")
        print(f"📝 Respuesta: {answer[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error RAG: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 PRUEBAS DEL SISTEMA RAG")
    print("=" * 50)
    
    # Cambiar al directorio correcto
    if not os.path.exists("articles"):
        print("❌ Ejecuta desde el directorio raíz del proyecto")
        return
    
    tests = [
        ("Imports básicos", test_basic_imports),
        ("Procesamiento PDF", test_pdf_processing),
        ("Groq API", test_groq_api),
        ("RAG simplificado", test_simple_rag)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        print(f"🧪 {test_name}")
        print(f"{'='*20}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print(f"\n{'='*50}")
    print("📊 RESUMEN DE PRUEBAS")
    print(f"{'='*50}")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n🎯 Resultado: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El sistema RAG debería funcionar.")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores arriba.")

if __name__ == "__main__":
    main()