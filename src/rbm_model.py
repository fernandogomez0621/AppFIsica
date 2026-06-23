"""
============================================================================
MÓDULO RBM - Máquina de Boltzmann Restringida
============================================================================

Implementación completa de Restricted Boltzmann Machine (RBM) para 
extracción de características latentes en datos de riesgo crediticio.

Características:
- Implementación desde cero del algoritmo RBM
- Contrastive Divergence (CD-k) para entrenamiento
- Métricas de evaluación completas
- Visualizaciones de diagnóstico
- Interfaz interactiva con Streamlit

Autor: Sistema de Física
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os
from typing import Tuple, Dict, List, Optional
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class RestrictedBoltzmannMachine:
    """
    Implementación completa de Máquina de Boltzmann Restringida (RBM)
    
    La RBM es un modelo generativo no supervisado que aprende representaciones
    latentes de los datos mediante una arquitectura de dos capas:
    - Capa visible: datos de entrada
    - Capa oculta: características latentes
    
    Entrenamiento mediante Contrastive Divergence (CD-k)
    """
    
    def __init__(self, 
                 n_visible: int,
                 n_hidden: int = 100,
                 learning_rate: float = 0.01,
                 n_epochs: int = 100,
                 batch_size: int = 64,
                 k_cd: int = 1,
                 random_state: int = 42):
        """
        Inicializa la RBM
        
        Args:
            n_visible: Número de unidades visibles (dimensión de entrada)
            n_hidden: Número de unidades ocultas
            learning_rate: Tasa de aprendizaje
            n_epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            k_cd: Número de pasos de Gibbs sampling en CD
            random_state: Semilla aleatoria
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k_cd = k_cd
        self.random_state = random_state
        
        # Inicializar generador aleatorio
        np.random.seed(random_state)
        
        # Inicializar parámetros
        self._initialize_parameters()
        
        # Métricas de entrenamiento
        self.training_history = {
            'reconstruction_error': [],
            'pseudo_log_likelihood': [],
            'free_energy': []
        }
        
        # Estado del modelo
        self.is_trained = False
        self.scaler = None
        
    def _initialize_parameters(self):
        """Inicializa pesos y sesgos de la RBM"""
        # Pesos: inicialización Xavier/Glorot
        std = np.sqrt(2.0 / (self.n_visible + self.n_hidden))
        self.W = np.random.normal(0, std, (self.n_visible, self.n_hidden))
        
        # Sesgos
        self.visible_bias = np.zeros(self.n_visible)
        self.hidden_bias = np.zeros(self.n_hidden)
        
        print(f"✓ Parámetros inicializados:")
        print(f"  - Pesos W: {self.W.shape}")
        print(f"  - Sesgo visible: {self.visible_bias.shape}")
        print(f"  - Sesgo oculto: {self.hidden_bias.shape}")
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Función sigmoide estable numéricamente"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    def _sample_hidden(self, visible: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Muestrea unidades ocultas dado el estado visible
        
        Args:
            visible: Estado de las unidades visibles
            
        Returns:
            hidden_probs: Probabilidades de activación de unidades ocultas
            hidden_states: Estados binarios de unidades ocultas
        """
        # Calcular probabilidades: P(h_j = 1 | v)
        hidden_probs = self._sigmoid(np.dot(visible, self.W) + self.hidden_bias)
        
        # Muestrear estados binarios
        hidden_states = (hidden_probs > np.random.random(hidden_probs.shape)).astype(np.float32)
        
        return hidden_probs, hidden_states
    
    def _sample_visible(self, hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Muestrea unidades visibles dado el estado oculto
        
        Args:
            hidden: Estado de las unidades ocultas
            
        Returns:
            visible_probs: Probabilidades de activación de unidades visibles
            visible_states: Estados de unidades visibles
        """
        # Calcular probabilidades: P(v_i = 1 | h)
        visible_probs = self._sigmoid(np.dot(hidden, self.W.T) + self.visible_bias)
        
        # Para datos continuos, usar las probabilidades directamente
        # Para datos binarios, muestrear estados binarios
        visible_states = visible_probs  # Asumiendo datos continuos normalizados
        
        return visible_probs, visible_states
    
    def _contrastive_divergence(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Implementa el algoritmo Contrastive Divergence (CD-k)
        
        Args:
            batch: Batch de datos de entrenamiento
            
        Returns:
            Gradientes para actualizar parámetros
        """
        batch_size = batch.shape[0]
        
        # Fase positiva
        pos_hidden_probs, pos_hidden_states = self._sample_hidden(batch)
        
        # Fase negativa: k pasos de Gibbs sampling
        neg_visible = batch.copy()
        for _ in range(self.k_cd):
            neg_hidden_probs, neg_hidden_states = self._sample_hidden(neg_visible)
            neg_visible_probs, neg_visible = self._sample_visible(neg_hidden_states)
        
        # Calcular gradientes
        pos_associations = np.dot(batch.T, pos_hidden_probs)
        neg_associations = np.dot(neg_visible.T, neg_hidden_probs)
        
        # Gradientes
        dW = (pos_associations - neg_associations) / batch_size
        dv_bias = np.mean(batch - neg_visible, axis=0)
        dh_bias = np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)
        
        return {
            'dW': dW,
            'dv_bias': dv_bias,
            'dh_bias': dh_bias,
            'reconstruction': neg_visible
        }
    
    def _compute_reconstruction_error(self, data: np.ndarray) -> float:
        """Calcula el error de reconstrucción"""
        hidden_probs, _ = self._sample_hidden(data)
        visible_probs, _ = self._sample_visible(hidden_probs)
        return mean_squared_error(data, visible_probs)
    
    def _compute_pseudo_log_likelihood(self, data: np.ndarray, n_samples: int = 100) -> float:
        """
        Calcula pseudo log-likelihood como aproximación de la verosimilitud
        
        Args:
            data: Datos de evaluación
            n_samples: Número de muestras para la estimación
            
        Returns:
            Pseudo log-likelihood promedio
        """
        if len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        pll = 0
        for sample in sample_data:
            # Calcular energía libre para el sample original
            fe_original = self._free_energy(sample.reshape(1, -1))
            
            # Para cada dimensión, calcular energía libre con bit flippeado
            fe_flipped = []
            for i in range(len(sample)):
                sample_flipped = sample.copy()
                sample_flipped[i] = 1 - sample_flipped[i]  # Flip bit
                fe_flipped.append(self._free_energy(sample_flipped.reshape(1, -1)))
            
            # Pseudo log-likelihood para este sample
            pll += self.n_visible * np.log(self._sigmoid(fe_flipped[0] - fe_original))
        
        return pll / len(sample_data)
    
    def _free_energy(self, visible: np.ndarray) -> float:
        """
        Calcula la energía libre: F(v) = -log(sum_h exp(-E(v,h)))
        
        Args:
            visible: Estado visible
            
        Returns:
            Energía libre
        """
        wx_b = np.dot(visible, self.W) + self.hidden_bias
        vbias_term = np.dot(visible, self.visible_bias)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term
    
    def fit(self, X: np.ndarray, validation_split: float = 0.2, verbose: bool = True) -> Dict:
        """
        Entrena la RBM usando Contrastive Divergence
        
        Args:
            X: Datos de entrenamiento
            validation_split: Proporción de datos para validación
            verbose: Si mostrar progreso
            
        Returns:
            Historia de entrenamiento
        """
        print(f"\n🚀 INICIANDO ENTRENAMIENTO RBM")
        print(f"{'='*50}")
        print(f"Arquitectura: {self.n_visible} → {self.n_hidden}")
        print(f"Hiperparámetros:")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Epochs: {self.n_epochs}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - CD steps: {self.k_cd}")
        print(f"{'='*50}")
        
        # Normalizar datos
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/validation
        if validation_split > 0:
            X_train, X_val = train_test_split(X_scaled, test_size=validation_split, 
                                            random_state=self.random_state)
        else:
            X_train = X_scaled
            X_val = None
        
        n_batches = len(X_train) // self.batch_size
        
        # Entrenamiento
        for epoch in range(self.n_epochs):
            epoch_error = 0
            
            # Shuffle datos
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            
            # Procesar batches
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch = X_train_shuffled[start_idx:end_idx]
                
                # Contrastive Divergence
                gradients = self._contrastive_divergence(batch)
                
                # Actualizar parámetros
                self.W += self.learning_rate * gradients['dW']
                self.visible_bias += self.learning_rate * gradients['dv_bias']
                self.hidden_bias += self.learning_rate * gradients['dh_bias']
                
                # Acumular error
                epoch_error += self._compute_reconstruction_error(batch)
            
            # Métricas de época
            avg_error = epoch_error / n_batches
            self.training_history['reconstruction_error'].append(avg_error)
            
            # Calcular métricas adicionales cada 10 épocas
            if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                if X_val is not None:
                    val_error = self._compute_reconstruction_error(X_val)
                    pll = self._compute_pseudo_log_likelihood(X_val[:100])  # Muestra pequeña
                    free_energy = np.mean(self._free_energy(X_val[:100]))
                    
                    self.training_history['pseudo_log_likelihood'].append(pll)
                    self.training_history['free_energy'].append(free_energy)
                    
                    if verbose:
                        print(f"Época {epoch+1:3d}/{self.n_epochs} | "
                              f"Error: {avg_error:.6f} | "
                              f"Val Error: {val_error:.6f} | "
                              f"PLL: {pll:.3f}")
                else:
                    if verbose:
                        print(f"Época {epoch+1:3d}/{self.n_epochs} | "
                              f"Error: {avg_error:.6f}")
        
        self.is_trained = True
        print(f"\n✅ ENTRENAMIENTO COMPLETADO")
        print(f"Error final: {self.training_history['reconstruction_error'][-1]:.6f}")
        
        return self.training_history
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extrae características de la capa oculta
        
        Args:
            X: Datos de entrada
            
        Returns:
            Activaciones de la capa oculta
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Normalizar usando el mismo scaler del entrenamiento
        X_scaled = self.scaler.transform(X)
        
        # Obtener activaciones ocultas
        hidden_probs, _ = self._sample_hidden(X_scaled)
        
        return hidden_probs
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruye datos desde la representación oculta
        
        Args:
            X: Datos de entrada
            
        Returns:
            Datos reconstruidos
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Normalizar
        X_scaled = self.scaler.transform(X)
        
        # Codificar y decodificar
        hidden_probs, _ = self._sample_hidden(X_scaled)
        visible_probs, _ = self._sample_visible(hidden_probs)
        
        # Desnormalizar
        reconstructed = self.scaler.inverse_transform(visible_probs)
        
        return reconstructed
    
    def generate_samples(self, n_samples: int = 100, n_gibbs: int = 1000) -> np.ndarray:
        """
        Genera muestras sintéticas usando Gibbs sampling
        
        Args:
            n_samples: Número de muestras a generar
            n_gibbs: Número de pasos de Gibbs sampling
            
        Returns:
            Muestras generadas
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Inicializar con ruido aleatorio
        samples = np.random.random((n_samples, self.n_visible))
        
        # Gibbs sampling
        for _ in range(n_gibbs):
            hidden_probs, hidden_states = self._sample_hidden(samples)
            visible_probs, samples = self._sample_visible(hidden_states)
        
        # Desnormalizar
        samples_denorm = self.scaler.inverse_transform(samples)
        
        return samples_denorm
    
    def save_model(self, filepath: str, feature_names: List[str] = None):
        """Guarda el modelo entrenado"""
        model_data = {
            'W': self.W,
            'visible_bias': self.visible_bias,
            'hidden_bias': self.hidden_bias,
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'k_cd': self.k_cd,
            'random_state': self.random_state,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Modelo guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Carga un modelo entrenado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Crear instancia
        rbm = cls(
            n_visible=model_data['n_visible'],
            n_hidden=model_data['n_hidden'],
            learning_rate=model_data['learning_rate'],
            n_epochs=model_data['n_epochs'],
            batch_size=model_data['batch_size'],
            k_cd=model_data['k_cd'],
            random_state=model_data['random_state']
        )
        
        # Restaurar parámetros
        rbm.W = model_data['W']
        rbm.visible_bias = model_data['visible_bias']
        rbm.hidden_bias = model_data['hidden_bias']
        rbm.training_history = model_data['training_history']
        rbm.is_trained = model_data['is_trained']
        rbm.scaler = model_data['scaler']
        rbm.feature_names = model_data.get('feature_names', None)
        
        print(f"✅ Modelo cargado desde: {filepath}")
        if rbm.feature_names:
            print(f"  - Características: {len(rbm.feature_names)}")
        return rbm

def create_rbm_visualizations(rbm: RestrictedBoltzmannMachine, 
                            X_original: np.ndarray,
                            feature_names: List[str] = None) -> Dict:
    """
    Crea visualizaciones de diagnóstico para la RBM
    
    Args:
        rbm: Modelo RBM entrenado
        X_original: Datos originales
        feature_names: Nombres de las características
        
    Returns:
        Diccionario con figuras de Plotly
    """
    figures = {}
    
    # 1. Curvas de aprendizaje
    if rbm.training_history['reconstruction_error']:
        fig_learning = go.Figure()
        
        epochs = list(range(1, len(rbm.training_history['reconstruction_error']) + 1))
        fig_learning.add_trace(go.Scatter(
            x=epochs,
            y=rbm.training_history['reconstruction_error'],
            mode='lines+markers',
            name='Error de Reconstrucción',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig_learning.update_layout(
            title="Curva de Aprendizaje - Error de Reconstrucción",
            xaxis_title="Época",
            yaxis_title="Error MSE",
            template="plotly_white",
            height=400
        )
        
        figures['learning_curve'] = fig_learning
    
    # 2. Heatmap de pesos
    fig_weights = go.Figure(data=go.Heatmap(
        z=rbm.W.T,
        colorscale='RdBu',
        zmid=0,
        showscale=True,
        colorbar=dict(title="Peso")
    ))
    
    fig_weights.update_layout(
        title=f"Matriz de Pesos W ({rbm.n_visible} × {rbm.n_hidden})",
        xaxis_title="Unidades Visibles",
        yaxis_title="Unidades Ocultas",
        template="plotly_white",
        height=500
    )
    
    figures['weights_heatmap'] = fig_weights
    
    # 3. Distribución de activaciones ocultas
    if rbm.is_trained:
        hidden_activations = rbm.transform(X_original)
        
        fig_activations = go.Figure()
        
        # Histograma de activaciones promedio por unidad oculta
        mean_activations = np.mean(hidden_activations, axis=0)
        
        fig_activations.add_trace(go.Histogram(
            x=mean_activations,
            nbinsx=30,
            name='Activaciones Promedio',
            marker_color='#3498db',
            opacity=0.7
        ))
        
        fig_activations.update_layout(
            title="Distribución de Activaciones de Unidades Ocultas",
            xaxis_title="Activación Promedio",
            yaxis_title="Frecuencia",
            template="plotly_white",
            height=400
        )
        
        figures['activations_dist'] = fig_activations
    
    # 4. Comparación Original vs Reconstruido
    if rbm.is_trained and len(X_original) > 0:
        # Tomar una muestra pequeña para visualización
        sample_size = min(100, len(X_original))
        sample_indices = np.random.choice(len(X_original), sample_size, replace=False)
        X_sample = X_original[sample_indices]
        
        X_reconstructed = rbm.reconstruct(X_sample)
        
        # Crear subplots para comparar algunas características
        n_features_to_show = min(6, X_original.shape[1])
        feature_indices = np.random.choice(X_original.shape[1], n_features_to_show, replace=False)
        
        fig_comparison = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"Característica {i}" for i in feature_indices],
            vertical_spacing=0.1
        )
        
        for idx, feat_idx in enumerate(feature_indices):
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            # Original
            fig_comparison.add_trace(
                go.Scatter(
                    x=list(range(len(X_sample))),
                    y=X_sample[:, feat_idx],
                    mode='markers',
                    name='Original',
                    marker=dict(color='#e74c3c', size=4),
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Reconstruido
            fig_comparison.add_trace(
                go.Scatter(
                    x=list(range(len(X_sample))),
                    y=X_reconstructed[:, feat_idx],
                    mode='markers',
                    name='Reconstruido',
                    marker=dict(color='#3498db', size=4),
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
        
        fig_comparison.update_layout(
            title="Comparación: Datos Originales vs Reconstruidos",
            template="plotly_white",
            height=600
        )
        
        figures['reconstruction_comparison'] = fig_comparison
    
    return figures

def render_rbm_module():
    """Renderiza el módulo completo de RBM en Streamlit"""
    st.title("⚡ Máquina de Boltzmann Restringida (RBM)")
    st.markdown("### *Extracción de características latentes mediante aprendizaje generativo*")
    
    # Información teórica
    with st.expander("📚 ¿Qué es una RBM?", expanded=False):
        st.markdown("""
        Una **Máquina de Boltzmann Restringida (RBM)** es un modelo generativo no supervisado 
        que aprende representaciones latentes de los datos.
        
        **Arquitectura:**
        - **Capa Visible:** Representa los datos de entrada
        - **Capa Oculta:** Captura características latentes
        - **Sin conexiones** dentro de cada capa (restricción)
        - **Conexiones bidireccionales** entre capas
        
        **Función de Energía:**
        ```
        E(v,h) = -∑ᵢ aᵢvᵢ - ∑ⱼ bⱼhⱼ - ∑ᵢⱼ vᵢWᵢⱼhⱼ
        ```
        
        **Entrenamiento:**
        - Algoritmo: **Contrastive Divergence (CD-k)**
        - Objetivo: Maximizar la verosimilitud de los datos
        - Aproximación: k pasos de Gibbs sampling
        """)
    
    # Verificar datos
    if not os.path.exists("data/processed/datos_credito_hipotecario_realista.csv"):
        st.error("❌ No hay datos disponibles. Ve a 'Generar Datos' primero.")
        return
    
    # Cargar datos
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/processed/datos_credito_hipotecario_realista.csv")
        return df
    
    df = load_data()
    st.success(f"✅ Datos cargados: {len(df):,} registros, {len(df.columns)} variables")
    
    # Configuración de la RBM
    st.subheader("⚙️ Configuración de la RBM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Arquitectura:**")
        n_hidden = st.slider(
            "Unidades ocultas",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Número de neuronas en la capa oculta"
        )
        
        learning_rate = st.select_slider(
            "Tasa de aprendizaje",
            options=[0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.01,
            help="Velocidad de actualización de parámetros"
        )
    
    with col2:
        st.markdown("**Entrenamiento:**")
        n_epochs = st.slider(
            "Épocas",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Número de iteraciones de entrenamiento"
        )
        
        batch_size = st.select_slider(
            "Tamaño de batch",
            options=[16, 32, 64, 128, 256],
            value=64,
            help="Número de muestras por batch"
        )
        
        k_cd = st.slider(
            "Pasos CD",
            min_value=1,
            max_value=10,
            value=1,
            help="Pasos de Gibbs sampling en Contrastive Divergence"
        )
    
    # Selección de características
    st.subheader("📊 Selección de Características")
    
    # Filtrar solo columnas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir algunas columnas problemáticas
    exclude_cols = ['nivel_riesgo_cod', 'rechazo_automatico'] if 'nivel_riesgo_cod' in numeric_columns else []
    numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
    
    selected_features = st.multiselect(
        "Selecciona las características para entrenar la RBM:",
        options=numeric_columns,
        default=numeric_columns[:15] if len(numeric_columns) > 15 else numeric_columns,
        help="Selecciona las variables numéricas para el entrenamiento"
    )
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una característica.")
        return
    
    # Preparar datos
    X = df[selected_features].values
    n_visible = len(selected_features)
    
    st.info(f"📊 Datos preparados: {X.shape[0]} muestras × {X.shape[1]} características")
    
    # Entrenamiento
    st.subheader("🚀 Entrenamiento de la RBM")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎯 Entrenar RBM", type="primary", use_container_width=True):
            with st.spinner("⏳ Entrenando Máquina de Boltzmann..."):
                try:
                    # Crear RBM
                    rbm = RestrictedBoltzmannMachine(
                        n_visible=n_visible,
                        n_hidden=n_hidden,
                        learning_rate=learning_rate,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        k_cd=k_cd,
                        random_state=42
                    )
                    
                    # Entrenar
                    history = rbm.fit(X, validation_split=0.2, verbose=False)
                    
                    # Guardar modelo con nombres de características
                    os.makedirs("models/rbm", exist_ok=True)
                    model_path = f"models/rbm/rbm_h{n_hidden}_lr{learning_rate}_e{n_epochs}.pkl"
                    rbm.save_model(model_path, feature_names=selected_features)
                    
                    # Guardar en session state
                    st.session_state.rbm_model = rbm
                    st.session_state.rbm_features = selected_features
                    st.session_state.rbm_data = X
                    
                    st.success("✅ RBM entrenada exitosamente!")
                    
                    # Mostrar métricas finales
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    with col_m1:
                        final_error = history['reconstruction_error'][-1]
                        st.metric("Error Reconstrucción", f"{final_error:.6f}")
                    
                    with col_m2:
                        if history['pseudo_log_likelihood']:
                            final_pll = float(history['pseudo_log_likelihood'][-1])
                            st.metric("Pseudo Log-Likelihood", f"{final_pll:.3f}")
                    
                    with col_m3:
                        if history['free_energy']:
                            final_fe = float(history['free_energy'][-1])
                            st.metric("Energía Libre", f"{final_fe:.3f}")
                    
                    with col_m4:
                        # Calcular sparsity de activaciones
                        hidden_act = rbm.transform(X[:1000])
                        sparsity = float(np.mean(hidden_act < 0.1))
                        st.metric("Sparsity", f"{sparsity:.2%}")
                    
                except Exception as e:
                    st.error(f"❌ Error durante el entrenamiento: {e}")
                    st.exception(e)
    
    with col2:
        # Cargar modelo existente
        model_files = []
        if os.path.exists("models/rbm"):
            model_files = [f for f in os.listdir("models/rbm") if f.endswith('.pkl')]
        
        if model_files:
            selected_model = st.selectbox(
                "O cargar modelo existente:",
                options=model_files,
                help="Selecciona un modelo RBM previamente entrenado"
            )
            
            if st.button("📂 Cargar Modelo", use_container_width=True):
                try:
                    model_path = f"models/rbm/{selected_model}"
                    rbm = RestrictedBoltzmannMachine.load_model(model_path)
                    
                    st.session_state.rbm_model = rbm
                    st.session_state.rbm_features = selected_features
                    st.session_state.rbm_data = X
                    
                    st.success("✅ Modelo cargado exitosamente!")
                    
                except Exception as e:
                    st.error(f"❌ Error cargando modelo: {e}")
    
    # Visualizaciones y análisis
    if 'rbm_model' in st.session_state:
        rbm = st.session_state.rbm_model
        
        st.divider()
        st.subheader("📊 Análisis y Visualizaciones")
        
        # Crear visualizaciones
        with st.spinner("🎨 Generando visualizaciones..."):
            figures = create_rbm_visualizations(rbm, X, selected_features)
        
        # Mostrar visualizaciones en tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Aprendizaje", 
            "🎨 Pesos", 
            "📊 Activaciones", 
            "🔄 Reconstrucción"
        ])
        
        with tab1:
            if 'learning_curve' in figures:
                st.plotly_chart(figures['learning_curve'], use_container_width=True)
            else:
                st.info("No hay datos de entrenamiento disponibles.")
        
        with tab2:
            if 'weights_heatmap' in figures:
                st.plotly_chart(figures['weights_heatmap'], use_container_width=True)
                
                st.markdown("""
                **Interpretación del Heatmap de Pesos:**
                - **Colores rojos:** Conexiones positivas (activación conjunta)
                - **Colores azules:** Conexiones negativas (inhibición)
                - **Patrones:** Indican qué características se agrupan
                """)
        
        with tab3:
            if 'activations_dist' in figures:
                st.plotly_chart(figures['activations_dist'], use_container_width=True)
                
                st.markdown("""
                **Análisis de Activaciones:**
                - Distribución de activaciones promedio de unidades ocultas
                - Valores cerca de 0.5 indican unidades balanceadas
                - Valores extremos (0 o 1) pueden indicar unidades especializadas
                """)
        
        with tab4:
            if 'reconstruction_comparison' in figures:
                st.plotly_chart(figures['reconstruction_comparison'], use_container_width=True)
                
                st.markdown("""
                **Calidad de Reconstrucción:**
                - Comparación entre datos originales y reconstruidos
                - Puntos cercanos indican buena reconstrucción
                - Dispersión indica pérdida de información
                """)
        
        # Extracción de características
        st.divider()
        st.subheader("🔧 Extracción de Características")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Extraer Características RBM", use_container_width=True):
                with st.spinner("⚡ Extrayendo características latentes..."):
                    # Extraer características
                    hidden_features = rbm.transform(X)
                    
                    # Crear DataFrame con características originales + RBM
                    feature_names_rbm = [f"RBM_H{i+1}" for i in range(hidden_features.shape[1])]
                    df_rbm = pd.DataFrame(hidden_features, columns=feature_names_rbm)
                    
                    # Combinar con datos originales
                    df_enhanced = pd.concat([df.reset_index(drop=True), df_rbm], axis=1)
                    
                    # Guardar dataset enriquecido
                    os.makedirs("data/processed", exist_ok=True)
                    enhanced_path = "data/processed/datos_con_rbm.csv"
                    df_enhanced.to_csv(enhanced_path, index=False)
                    
                    st.success(f"✅ Características extraídas: {hidden_features.shape[1]} nuevas variables")
                    st.success(f"💾 Dataset enriquecido guardado: {enhanced_path}")
                    
                    # Mostrar estadísticas
                    st.metric("Características RBM", hidden_features.shape[1])
                    st.metric("Dataset Total", f"{df_enhanced.shape[1]} variables")
        
        with col2:
            if st.button("🎲 Generar Muestras Sintéticas", use_container_width=True):
                with st.spinner("🎲 Generando muestras sintéticas..."):
                    try:
                        # Generar muestras
                        synthetic_samples = rbm.generate_samples(n_samples=100, n_gibbs=1000)
                        
                        # Crear DataFrame
                        df_synthetic = pd.DataFrame(synthetic_samples, columns=selected_features)
                        
                        # Guardar
                        os.makedirs("data/synthetic", exist_ok=True)
                        synthetic_path = "data/synthetic/rbm_synthetic_samples.csv"
                        df_synthetic.to_csv(synthetic_path, index=False)
                        
                        st.success("✅ Muestras sintéticas generadas!")
                        st.success(f"💾 Guardadas en: {synthetic_path}")
                        
                        # Mostrar muestra
                        st.dataframe(df_synthetic.head(), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error generando muestras: {e}")

if __name__ == "__main__":
    # Para testing
    print("Módulo RBM cargado correctamente")