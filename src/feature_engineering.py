"""
============================================================================
MÓDULO DE INGENIERÍA DE CARACTERÍSTICAS
============================================================================

Creación automática de variables derivadas para mejorar el poder predictivo
de los modelos de riesgo crediticio.

Autor: Sistema de Física
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from typing import Dict, List, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Ingeniero de características para datos de crédito hipotecario"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el ingeniero de características
        
        Args:
            data: DataFrame con datos originales
        """
        self.data = data.copy()
        self.original_columns = data.columns.tolist()
        self.new_features = {}
        self.feature_importance = {}
        
    def create_financial_ratios(self) -> pd.DataFrame:
        """Crea ratios financieros fundamentales"""
        df = self.data.copy()
        
        # 1. Loan-to-Value (LTV) - Ya existe pero verificamos
        if 'valor_inmueble' in df.columns and 'monto_credito' in df.columns:
            df['ltv_ratio'] = (df['monto_credito'] / df['valor_inmueble']) * 100
            self.new_features['ltv_ratio'] = "Ratio Préstamo/Valor del inmueble (%)"
        
        # 2. Debt-to-Income (DTI) - Ya existe pero verificamos
        if 'cuota_mensual' in df.columns and 'salario_mensual' in df.columns:
            df['dti_ratio'] = (df['cuota_mensual'] / df['salario_mensual']) * 100
            self.new_features['dti_ratio'] = "Ratio Deuda/Ingreso (%)"
        
        # 3. Capacidad de ahorro
        if 'salario_mensual' in df.columns and 'egresos_mensuales' in df.columns:
            df['capacidad_ahorro_nueva'] = df['salario_mensual'] - df['egresos_mensuales']
            df['ratio_ahorro_salario'] = (df['capacidad_ahorro_nueva'] / df['salario_mensual']) * 100
            self.new_features['capacidad_ahorro_nueva'] = "Capacidad de ahorro mensual (COP)"
            self.new_features['ratio_ahorro_salario'] = "Ratio Ahorro/Salario (%)"
        
        # 4. Ratio patrimonio/deuda
        if 'patrimonio_total' in df.columns and 'monto_credito' in df.columns:
            df['ratio_patrimonio_deuda'] = df['patrimonio_total'] / (df['monto_credito'] + 1)
            self.new_features['ratio_patrimonio_deuda'] = "Ratio Patrimonio/Deuda"
        
        # 5. Saldo relativo
        if 'saldo_promedio_banco' in df.columns and 'salario_mensual' in df.columns:
            df['saldo_relativo'] = df['saldo_promedio_banco'] / (df['salario_mensual'] + 1)
            self.new_features['saldo_relativo'] = "Saldo banco relativo al salario"
        
        # 6. Meses de colchón financiero
        if 'saldo_promedio_banco' in df.columns and 'cuota_mensual' in df.columns:
            df['meses_colchon'] = df['saldo_promedio_banco'] / (df['cuota_mensual'] + 1)
            self.new_features['meses_colchon'] = "Meses de colchón financiero"
        
        # 7. Ratio cuota inicial
        if 'valor_cuota_inicial' in df.columns and 'valor_inmueble' in df.columns:
            df['ratio_cuota_inicial'] = (df['valor_cuota_inicial'] / df['valor_inmueble']) * 100
            self.new_features['ratio_cuota_inicial'] = "Ratio Cuota Inicial (%)"
        
        
        # NUEVOS RATIOS PARA COMPATIBILIDAD CON RBM
        # 8. Ratio cuota/ingreso
        if 'cuota_mensual' in df.columns and 'salario_mensual' in df.columns:
            df['ratio_cuota_ingreso'] = df['cuota_mensual'] / (df['salario_mensual'] + 1)
            self.new_features['ratio_cuota_ingreso'] = "Ratio Cuota/Ingreso"
        
        # 9. Ratio cuota/ahorro
        if 'cuota_mensual' in df.columns and 'capacidad_ahorro' in df.columns:
            df['ratio_cuota_ahorro'] = df['cuota_mensual'] / (df['capacidad_ahorro'] + 1)
            self.new_features['ratio_cuota_ahorro'] = "Ratio Cuota/Ahorro"
        
        # 10. Ratio egreso/salario
        if 'egresos_mensuales' in df.columns and 'salario_mensual' in df.columns:
            df['ratio_egreso_salario'] = df['egresos_mensuales'] / (df['salario_mensual'] + 1)
            self.new_features['ratio_egreso_salario'] = "Ratio Egreso/Salario"
        return df
    
    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea indicadores de riesgo específicos"""
        
        # 1. Score de edad (penalización por edades extremas)
        if 'edad' in df.columns:
            df['score_edad'] = df['edad'].apply(self._calculate_age_score)
            self.new_features['score_edad'] = "Score de riesgo por edad"
        
        # 2. Indicador de sobreendeudamiento
        if 'dti_ratio' in df.columns:
            df['flag_sobreendeudamiento'] = (df['dti_ratio'] > 40).astype(int)
            df['nivel_sobreendeudamiento'] = pd.cut(
                df['dti_ratio'],
                bins=[0, 25, 35, 45, 100],
                labels=['Bajo', 'Moderado', 'Alto', 'Crítico']
            )
            self.new_features['flag_sobreendeudamiento'] = "Flag sobreendeudamiento (DTI > 40%)"
            self.new_features['nivel_sobreendeudamiento'] = "Nivel de sobreendeudamiento"
        
        # 3. Score de estabilidad laboral (AMBOS NOMBRES para compatibilidad)
        if 'antiguedad_empleo' in df.columns and 'tipo_empleo' in df.columns:
            df['score_estabilidad'] = df.apply(self._calculate_stability_score, axis=1)
            df['score_estabilidad_laboral'] = df['score_estabilidad']  # Alias para RBM
            self.new_features['score_estabilidad'] = "Score de estabilidad laboral"
            self.new_features['score_estabilidad_laboral'] = "Score de estabilidad laboral (alias)"
        
        # 4. Riesgo legal (función exponencial de demandas)
        if 'numero_demandas' in df.columns:
            df['riesgo_legal'] = 100 * (1 - np.exp(-2 * df['numero_demandas']))
            self.new_features['riesgo_legal'] = "Riesgo legal (% basado en demandas)"
        
        # 5. Score de educación (ordinal)
        if 'nivel_educacion' in df.columns:
            education_map = {
                'Bachiller': 1,
                'Técnico': 2,
                'Profesional': 3,
                'Posgrado': 4
            }
            df['score_educacion'] = df['nivel_educacion'].map(education_map)
            self.new_features['score_educacion'] = "Score ordinal de educación"
        
        # 6. Indicador de alta liquidez
        if 'saldo_promedio_banco' in df.columns and 'salario_mensual' in df.columns:
            df['flag_alta_liquidez'] = (df['saldo_promedio_banco'] > df['salario_mensual'] * 3).astype(int)
            self.new_features['flag_alta_liquidez'] = "Flag alta liquidez (saldo > 3 salarios)"
        
        # 7. Puntaje de riesgo compuesto (para compatibilidad con RBM)
        if 'puntaje_datacredito' in df.columns and 'dti_ratio' in df.columns:
            # Normalizar puntaje datacredito (0-100)
            puntaje_norm = (df['puntaje_datacredito'] - 150) / (950 - 150) * 100
            # Invertir DTI (menor DTI = mejor puntaje)
            dti_inv = 100 - df['dti_ratio'].clip(0, 100)
            # Combinar (60% datacredito, 40% DTI)
            df['puntaje_riesgo'] = 0.6 * puntaje_norm + 0.4 * dti_inv
            self.new_features['puntaje_riesgo'] = "Puntaje de riesgo compuesto (0-100)"
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea variables de interacción"""
        
        # 1. Educación × Salario
        if 'score_educacion' in df.columns and 'salario_mensual' in df.columns:
            df['educacion_x_salario'] = df['score_educacion'] * (df['salario_mensual'] / 1000000)
            self.new_features['educacion_x_salario'] = "Interacción Educación × Salario"
        
        # 2. Propiedades × Patrimonio
        if 'numero_propiedades' in df.columns and 'patrimonio_total' in df.columns:
            df['propiedades_x_patrimonio'] = df['numero_propiedades'] * np.log(df['patrimonio_total'] + 1)
            self.new_features['propiedades_x_patrimonio'] = "Interacción Propiedades × Log(Patrimonio)"
        
        # 3. Edad × Empleo (AMBOS NOMBRES para compatibilidad)
        if 'edad' in df.columns and 'antiguedad_empleo' in df.columns:
            df['edad_x_empleo'] = df['edad'] * df['antiguedad_empleo']
            df['edad_x_antiguedad'] = df['edad_x_empleo']  # Alias para RBM
            self.new_features['edad_x_empleo'] = "Interacción Edad × Antigüedad Empleo"
            self.new_features['edad_x_antiguedad'] = "Interacción Edad × Antigüedad (alias)"
        
        # 4. LTV × Puntaje DataCrédito
        if 'ltv_ratio' in df.columns and 'puntaje_datacredito' in df.columns:
            df['ltv_x_puntaje'] = df['ltv_ratio'] * (900 - df['puntaje_datacredito']) / 100
            self.new_features['ltv_x_puntaje'] = "Interacción LTV × (900 - Puntaje DataCrédito)"
        
        return df
    
    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea variables discretizadas/binned"""
        
        # 1. Grupos de edad
        if 'edad' in df.columns:
            df['grupo_edad'] = pd.cut(
                df['edad'],
                bins=[0, 30, 40, 55, 100],
                labels=['Joven', 'Adulto_Joven', 'Adulto', 'Adulto_Mayor']
            )
            self.new_features['grupo_edad'] = "Grupo etario"
        
        # 2. Rangos salariales
        if 'salario_mensual' in df.columns:
            df['rango_salarial'] = pd.cut(
                df['salario_mensual'],
                bins=[0, 2000000, 3500000, 5000000, 8000000, np.inf],
                labels=['Muy_Bajo', 'Bajo', 'Medio', 'Alto', 'Muy_Alto']
            )
            self.new_features['rango_salarial'] = "Rango salarial"
        
        # 3. Categorías de puntaje DataCrédito
        if 'puntaje_datacredito' in df.columns:
            df['categoria_puntaje'] = pd.cut(
                df['puntaje_datacredito'],
                bins=[0, 500, 600, 700, 800, 950],
                labels=['Malo', 'Regular', 'Bueno', 'Muy_Bueno', 'Excelente']
            )
            self.new_features['categoria_puntaje'] = "Categoría puntaje DataCrédito"
        
        # 4. Niveles de LTV
        if 'ltv_ratio' in df.columns:
            df['nivel_ltv'] = pd.cut(
                df['ltv_ratio'],
                bins=[0, 60, 70, 80, 90, 100],
                labels=['Muy_Bajo', 'Bajo', 'Medio', 'Alto', 'Muy_Alto']
            )
            self.new_features['nivel_ltv'] = "Nivel de LTV"
        
        return df
    
    def create_transformed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea variables transformadas matemáticamente"""
        
        # Variables con distribución sesgada para transformar con log
        skewed_vars = ['salario_mensual', 'patrimonio_total', 'valor_inmueble', 'saldo_promedio_banco']
        
        for var in skewed_vars:
            if var in df.columns:
                # Log transformation
                df[f'{var}_log'] = np.log(df[var] + 1)
                self.new_features[f'{var}_log'] = f"Log({var})"
                
                # Square root transformation
                df[f'{var}_sqrt'] = np.sqrt(df[var])
                self.new_features[f'{var}_sqrt'] = f"Raíz({var})"
        
        # Transformaciones específicas
        if 'dti_ratio' in df.columns:
            df['dti_cuadrado'] = df['dti_ratio'] ** 2
            self.new_features['dti_cuadrado'] = "DTI al cuadrado"
        
        if 'edad' in df.columns:
            df['edad_cuadrado'] = df['edad'] ** 2
            self.new_features['edad_cuadrado'] = "Edad al cuadrado"
        
        return df
    
    def _calculate_age_score(self, age: float) -> float:
        """Calcula score de riesgo por edad"""
        if age < 25:
            return -30  # Penalización por juventud
        elif age < 30:
            return 10
        elif age <= 55:
            return 40   # Edad óptima
        else:
            return max(-100, -8 * (age - 55))  # Penalización por edad avanzada
    
    def _calculate_stability_score(self, row) -> float:
        """Calcula score de estabilidad laboral"""
        antiguedad = row['antiguedad_empleo']
        tipo_empleo = row['tipo_empleo']
        
        # Score base por antigüedad
        score = min(100, antiguedad * 10)
        
        # Ajuste por tipo de empleo
        if tipo_empleo == "Formal":
            score += 25
        elif tipo_empleo == "Independiente":
            score += 10
        # Informal no suma puntos
        
        return max(0, min(125, score))
    
    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str = 'nivel_riesgo') -> Dict:
        """
        Calcula importancia de características usando mutual information
        
        Args:
            df: DataFrame con características
            target_col: Variable objetivo
            
        Returns:
            Diccionario con importancias
        """
        if target_col not in df.columns:
            return {}
        
        # Preparar datos
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_features:
            numeric_features.remove(target_col)
        
        X = df[numeric_features].fillna(0)
        
        # Codificar target si es categórico
        if df[target_col].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(df[target_col].fillna('Unknown'))
        else:
            y = df[target_col].fillna(0)
        
        # Calcular mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Crear diccionario de importancias
        importance_dict = dict(zip(numeric_features, mi_scores))
        
        # Ordenar por importancia
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def generate_all_features(self) -> pd.DataFrame:
        """Genera todas las características derivadas"""
        
        print("🔧 INICIANDO INGENIERÍA DE CARACTERÍSTICAS")
        print("=" * 50)
        
        df = self.data.copy()
        initial_features = len(df.columns)
        
        # 1. Ratios financieros
        print("💰 Creando ratios financieros...")
        df = self.create_financial_ratios()
        
        # 2. Indicadores de riesgo
        print("⚠️ Creando indicadores de riesgo...")
        df = self.create_risk_indicators(df)
        
        # 3. Variables de interacción
        print("🔗 Creando variables de interacción...")
        df = self.create_interaction_features(df)
        
        # 4. Variables discretizadas
        print("📊 Creando variables discretizadas...")
        df = self.create_binned_features(df)
        
        # 5. Transformaciones matemáticas
        print("📐 Aplicando transformaciones matemáticas...")
        df = self.create_transformed_features(df)
        
        final_features = len(df.columns)
        new_features_count = final_features - initial_features
        
        print(f"✅ Ingeniería completada:")
        print(f"  - Características originales: {initial_features}")
        print(f"  - Características nuevas: {new_features_count}")
        print(f"  - Total características: {final_features}")
        print("=" * 50)
        
        return df

def render_feature_engineering():
    """Renderiza el módulo de ingeniería de características en Streamlit"""
    st.title("🔧 Ingeniería de Características")
    st.markdown("### *Creación automática de variables derivadas*")
    
    # Verificar datos
    if not os.path.exists("data/processed/datos_credito_hipotecario_realista.csv"):
        st.error("❌ No hay datos disponibles. Ve a 'Generar Datos' primero.")
        return
    
    # Cargar datos
    @st.cache_data
    def load_data():
        return pd.read_csv("data/processed/datos_credito_hipotecario_realista.csv")
    
    df = load_data()
    st.success(f"✅ Datos cargados: {len(df):,} registros, {len(df.columns)} variables originales")
    
    # Información sobre características a crear
    with st.expander("📋 Características que se van a crear", expanded=True):
        st.markdown("""
        ### 💰 Ratios Financieros:
        - `ltv_ratio`: Loan-to-Value ratio (%)
        - `dti_ratio`: Debt-to-Income ratio (%)
        - `ratio_ahorro_salario`: Capacidad ahorro/salario (%)
        - `ratio_patrimonio_deuda`: Patrimonio/Deuda
        - `saldo_relativo`: Saldo banco/salario
        - `meses_colchon`: Meses de colchón financiero
        
        ### ⚠️ Indicadores de Riesgo:
        - `score_edad`: Penalización por edades extremas
        - `flag_sobreendeudamiento`: DTI > 40%
        - `score_estabilidad`: Estabilidad laboral
        - `riesgo_legal`: Función exponencial de demandas
        - `score_educacion`: Codificación ordinal educación
        
        ### 🔗 Variables de Interacción:
        - `educacion_x_salario`: Educación × Salario
        - `propiedades_x_patrimonio`: Propiedades × Log(Patrimonio)
        - `edad_x_empleo`: Edad × Antigüedad empleo
        
        ### 📊 Variables Discretizadas:
        - `grupo_edad`: Joven/Adulto/Mayor
        - `rango_salarial`: Bajo/Medio/Alto
        - `categoria_puntaje`: Malo/Regular/Bueno/Excelente
        
        ### 📐 Transformaciones:
        - Variables log y raíz cuadrada para distribuciones sesgadas
        """)
    
    # Botón para generar características
    if st.button("🚀 Generar Características", type="primary", use_container_width=True):
        with st.spinner("🔧 Creando características derivadas..."):
            try:
                # Crear ingeniero
                engineer = FeatureEngineer(df)
                
                # Generar todas las características
                df_enhanced = engineer.generate_all_features()
                
                # Calcular importancia de características
                if 'nivel_riesgo' in df_enhanced.columns:
                    importance_scores = engineer.calculate_feature_importance(df_enhanced)
                else:
                    importance_scores = {}
                
                # Guardar dataset enriquecido
                os.makedirs("data/processed", exist_ok=True)
                enhanced_path = "data/processed/datos_con_caracteristicas.csv"
                df_enhanced.to_csv(enhanced_path, index=False)
                
                st.success(f"✅ Características creadas exitosamente!")
                st.success(f"💾 Dataset enriquecido guardado: {enhanced_path}")
                
                # Mostrar estadísticas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Características Originales", len(df.columns))
                
                with col2:
                    new_features_count = len(df_enhanced.columns) - len(df.columns)
                    st.metric("Características Nuevas", new_features_count)
                
                with col3:
                    st.metric("Total Características", len(df_enhanced.columns))
                
                # Mostrar nuevas características creadas
                st.subheader("📋 Nuevas Características Creadas")
                
                new_features_df = pd.DataFrame([
                    [feature, description]
                    for feature, description in engineer.new_features.items()
                ], columns=["Característica", "Descripción"])
                
                st.dataframe(new_features_df, use_container_width=True, hide_index=True)
                
                # Mostrar importancia de características
                if importance_scores:
                    st.subheader("📊 Importancia de Características")
                    
                    # Top 20 características más importantes
                    top_features = list(importance_scores.items())[:20]
                    
                    if top_features:
                        features_names = [f[0] for f in top_features]
                        importance_values = [f[1] for f in top_features]
                        
                        fig_importance = px.bar(
                            x=importance_values,
                            y=features_names,
                            orientation='h',
                            title="Top 20 Características Más Importantes",
                            labels={'x': 'Importancia (Mutual Information)', 'y': 'Características'}
                        )
                        
                        fig_importance.update_layout(
                            template="plotly_white",
                            height=600,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                # Mostrar muestra del dataset enriquecido
                st.subheader("👀 Vista Previa del Dataset Enriquecido")
                
                # Mostrar solo algunas columnas nuevas para no saturar
                new_cols = list(engineer.new_features.keys())[:10]
                display_cols = ['edad', 'salario_mensual', 'puntaje_datacredito', 'nivel_riesgo'] + new_cols
                display_cols = [col for col in display_cols if col in df_enhanced.columns]
                
                st.dataframe(
                    df_enhanced[display_cols].head(10),
                    use_container_width=True
                )
                
                # Botón de descarga
                csv = df_enhanced.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Dataset Enriquecido",
                    data=csv,
                    file_name="datos_con_caracteristicas.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error creando características: {e}")
                st.exception(e)
    
    # Mostrar características existentes si ya fueron creadas
    enhanced_path = "data/processed/datos_con_caracteristicas.csv"
    if os.path.exists(enhanced_path):
        st.divider()
        st.subheader("📊 Dataset Enriquecido Existente")
        
        try:
            df_existing = pd.read_csv(enhanced_path)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Registros", f"{len(df_existing):,}")
            with col2:
                st.metric("Total Características", len(df_existing.columns))
            with col3:
                nuevas = len(df_existing.columns) - len(df.columns)
                st.metric("Características Añadidas", nuevas)
            
            # Mostrar muestra
            st.dataframe(df_existing.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error cargando dataset enriquecido: {e}")

def render_feature_engineering_module():
    """Función principal para renderizar el módulo de ingeniería"""
    render_feature_engineering()

if __name__ == "__main__":
    print("Módulo de ingeniería de características cargado correctamente")