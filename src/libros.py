#!/usr/bin/env python3
"""
Script para descargar automáticamente papers sobre RBMs
"""

import os
import requests
import arxiv
from pathlib import Path

def download_papers():
    """Descarga papers clave sobre RBMs"""
    
    # Crear directorio
    papers_dir = Path("./articles/")
    papers_dir.mkdir(exist_ok=True)
    
    print("📚 Descargando papers sobre Máquinas de Boltzmann...")
    print("=" * 60)
    
    # ============================================
    # PAPERS DE URLs DIRECTAS
    # ============================================
    direct_urls = {
        "Hinton_2002_Contrastive_Divergence.pdf": 
            "https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf",
        
        "Hinton_Salakhutdinov_2006_Science.pdf": 
            "https://www.cs.toronto.edu/~hinton/absps/science.pdf",
        
        "Hinton_2010_Practical_Guide_RBMs.pdf": 
            "https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf",
        
        "Salakhutdinov_2007_Collaborative_Filtering.pdf": 
            "https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf",
        
        "Larochelle_2008_Discriminative_RBMs.pdf":
            "https://www.cs.toronto.edu/~larocheh/publications/icml-2008-discriminative-rbm.pdf",
        
        "Ackley_1985_Boltzmann_Machines.pdf":
            "https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf",
        
        "Tieleman_2008_PCD.pdf":
            "https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf",
        
        "Salakhutdinov_2009_Deep_BMs.pdf":
            "http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf",
    }
    
    print("\n📥 Descargando desde URLs directas...")
    for filename, url in direct_urls.items():
        filepath = papers_dir / filename
        
        if filepath.exists():
            print(f"⏭️  {filename} (ya existe)")
            continue
        
        try:
            response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ {filename}")
        
        except Exception as e:
            print(f"❌ Error descargando {filename}: {e}")
    
    # ============================================
    # PAPERS DE ARXIV
    # ============================================
    arxiv_papers = [
        ("1206.5538", "Fischer_Igel_2012_Introduction_RBMs.pdf"),
        ("1207.4404", "Bengio_2013_Better_Mixing.pdf"),
        ("1105.6169", "Montufar_2011_Universal_Approximation.pdf"),
        ("1312.6114", "Kingma_2014_VAE.pdf"),
        ("1406.2661", "Goodfellow_2014_GANs.pdf"),
    ]
    
    print("\n📥 Descargando desde arXiv...")
    for arxiv_id, filename in arxiv_papers:
        filepath = papers_dir / filename
        
        if filepath.exists():
            print(f"⏭️  {filename} (ya existe)")
            continue
        
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            paper.download_pdf(filename=str(filepath))
            print(f"✅ {filename}")
        
        except Exception as e:
            print(f"❌ Error descargando {arxiv_id}: {e}")
    
    # ============================================
    # RESUMEN FINAL
    # ============================================
    print("\n" + "=" * 60)
    total_papers = len(list(papers_dir.glob("*.pdf")))
    print(f"🎉 Descarga completa: {total_papers} papers en {papers_dir}")
    print("\n💡 Ahora ejecuta la app de Streamlit:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import arxiv
        import requests
    except ImportError:
        print("❌ Faltan dependencias. Instala con:")
        print("   pip install arxiv requests")
        exit(1)
    
    download_papers()