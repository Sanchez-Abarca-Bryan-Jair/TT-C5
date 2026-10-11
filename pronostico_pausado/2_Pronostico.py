import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from scipy.spatial import cKDTree

from utils.carga_datos import cargar_incidentes
from utils.modelo import (
    cargar_artefactos,
    construir_ventana_reciente,
    predecir_riesgo,
    artefactos_disponibles,
)

# ==========================================
# MEJORA VISUAL 1: BANNER ESTILO "PANORAMA"
# ==========================================
banner_html = """
<div style="background: linear-gradient(90deg, #902441 0%, #202b46 100%); padding: 25px; border-radius: 10px; margin-bottom: 25px;">
    <h1 style="color: white; font-size: 2.2rem; margin-top: 0; margin-bottom: 10px;">🔮 Pronóstico y Asignación Heurística</h1>
    <p style="color: #e0e0e0; font-size: 1.1rem; margin: 0;">Proyección espacial de siniestros viales en Iztapalapa mediante GCN-LSTM</p>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

# ==========================================
# MEJORA VISUAL 2: FUNCIÓN PARA SUBTÍTULOS
# ==========================================
def subtitulo_con_estilo(texto):
    st.markdown(f"""
    <div style="border-left: 6px solid #902441; padding-left: 12px; margin-top: 30px; margin-bottom: 15px;">
        <h3 style="margin: 0; color: #1F294C; font-size: 1.5rem;">{texto}</h3>
    </div>
    """, unsafe_allow_html=True)

st.info(
    "**Versión de Producción (TT):** La ventana de entrada utiliza las últimas 24 horas "
    "históricas para el pronóstico espacial. La topología está discretizada visualmente."
)

if not artefactos_disponibles():
    st.error("No se encontraron los artefactos del modelo en data/")
    st.stop()

# 1. Cargar Datos y Modelo
modelo, nodos_df, edge_index, metadatos = cargar_artefactos()
df_incidentes = cargar_incidentes()
T = metadatos.get("T", 24)

# 2. Inferencia
matriz_ventana = construir_ventana_reciente(df_incidentes, nodos_df, T, metadatos.get("n_features", 2))
probs = predecir_riesgo(modelo, matriz_ventana, edge_index)
nodos_df["probabilidad_riesgo"] = probs

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("### Ajustes de Operación")
umbral = st.sidebar.slider(
    "Umbral crítico de riesgo",
    min_value=0.0, max_value=1.0, value=0.75, step=0.05,
    help="Filtra los nodos que superen esta probabilidad de siniestro."
)

df_critico = nodos_df[nodos_df["probabilidad_riesgo"] >= umbral].copy()

if df_critico.empty:
    st.success(f"No hay zonas con riesgo superior al {umbral*100:.0f}% para la siguiente hora.")
    st.stop()

# ==========================================
# SECCIÓN: MAPA DE RIESGO
# ==========================================
subtitulo_con_estilo(f"Mapa de Riesgo Espacial (Umbral > {umbral*100:.0f}%)")

# Opciones de visualización (Segmented Control)
opcion_mapa = st.pills(
    "Estilo de visualización",
    options=["Original", "Hexágonos (2D)", "Elevación (3D)"],
    default="Elevación (3D)"
)

vista_inicial = pdk.ViewState(
    latitude=19.345, 
    longitude=-99.065,
    zoom=11.5,
    pitch=45 if opcion_mapa == "Elevación (3D)" else 0,
)

if opcion_mapa == "Original":
    capa_mapa = pdk.Layer(
        "ScatterplotLayer",
        data=df_critico,
        get_position=["lon", "lat"],
        get_color=[227, 26, 28, 160], 
        get_radius=80,
        pickable=True,
    )
    tooltip_texto = "Nodo de Riesgo"
else:
    es_3d = (opcion_mapa == "Elevación (3D)")
    capa_mapa = pdk.Layer(
        "HexagonLayer",
        data=df_critico,
        get_position=["lon", "lat"],
        radius=150, 
        elevation_scale=50 if es_3d else 0,
        elevation_range=[0, 1000],
        extruded=es_3d,
        get_elevation_weight="probabilidad_riesgo",
        color_range=[
            [254, 217, 118], 
            [253, 141, 60],  
            [227, 26, 28],   
            [128, 0, 38]     
        ],
        pickable=True,
    )
    tooltip_texto = "Concentración de Riesgo Crítico en esta zona"

st.pydeck_chart(pdk.Deck(
    layers=[capa_mapa],
    initial_view_state=vista_inicial,
    tooltip={"text": tooltip_texto}
))

# ==========================================
# SECCIÓN: TOP NODOS CRÍTICOS
# ==========================================
subtitulo_con_estilo("Top Nodos Críticos")

# Hacemos una copia y escalamos la probabilidad a porcentaje (0-100) para mejor lectura
df_mostrar = df_critico.sort_values(by="probabilidad_riesgo", ascending=False).head(20).copy()
df_mostrar["probabilidad_pct"] = df_mostrar["probabilidad_riesgo"] * 100

st.dataframe(
    df_mostrar[["nodo_id", "lat", "lon", "probabilidad_pct"]],
    column_config={
        "nodo_id": "ID Nodo Vial",
        "lat": st.column_config.NumberColumn("Latitud", format="%.4f"),
        "lon": st.column_config.NumberColumn("Longitud", format="%.4f"),
        "probabilidad_pct": st.column_config.ProgressColumn(
            "Nivel de Riesgo",
            help="Probabilidad calculada por la red GCN-LSTM",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
    },
    hide_index=True,
    use_container_width=True
)

# ==========================================
# SECCIÓN: ASIGNACIÓN LOGÍSTICA DE DOBLE TRAMO
# ==========================================
subtitulo_con_estilo("Asignación Logística Integral (Doble Tramo)")

st.markdown("Cálculo heurístico de **Ruta Completa**: Tiempo de arribo de la unidad C5 + traslado a centro de trauma.")

# 1. Catálogos operativos
bases_c5 = pd.DataFrame({
    "nombre_base": ["Sector Abasto", "Sector Santa Cruz", "Sector Cabeza de Juárez"],
    "lat": [19.3738, 19.3400, 19.3800],
    "lon": [-99.0953, -99.0300, -99.0400]
})

hospitales_trauma = pd.DataFrame({
    "nombre_hospital": ["Hospital General Iztapalapa", "Hospital Belisario Domínguez", "Clínica 47 IMSS"],
    "lat": [19.3446, 19.3142, 19.3361],
    "lon": [-99.0286, -99.0664, -99.0383]
})

# 2. Heurística de Proximidad mediante KD-Tree
arbol_bases = cKDTree(bases_c5[["lat", "lon"]].values)
arbol_hospitales = cKDTree(hospitales_trauma[["lat", "lon"]].values)

top_5_nodos = df_mostrar.head(5).copy()

# Tramo A: Base C5 al Siniestro
dist_base, idx_base = arbol_bases.query(top_5_nodos[["lat", "lon"]].values)
top_5_nodos["base_c5"] = bases_c5.iloc[idx_base]["nombre_base"].values
top_5_nodos["t_respuesta_min"] = (dist_base * 111.0 / 30.0) * 60 

# Tramo B: Siniestro al Hospital
dist_hosp, idx_hosp = arbol_hospitales.query(top_5_nodos[["lat", "lon"]].values)
top_5_nodos["hospital"] = hospitales_trauma.iloc[idx_hosp]["nombre_hospital"].values
top_5_nodos["t_traslado_min"] = (dist_hosp * 111.0 / 30.0) * 60 

# Tiempo Total Operativo
top_5_nodos["tiempo_total_min"] = top_5_nodos["t_respuesta_min"] + top_5_nodos["t_traslado_min"]

# 3. Tarjetas de Alerta Máxima

nodo_top_1 = top_5_nodos.iloc[0]

st.error("**ALERTA MÉDICA PRIORITARIA:** Riesgo inminente con posible requerimiento de traslado hospitalario.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Riesgo (Nodo)", f"{nodo_top_1['probabilidad_pct']:.1f}%")
with col2:
    st.metric("Despacho Sugerido", nodo_top_1['base_c5'])
with col3:
    st.metric("Recepción Médica", nodo_top_1['hospital'])
with col4:
    st.metric("Tiempo Total de Operación", f"{nodo_top_1['tiempo_total_min']:.1f} min", delta="- Vía Óptima", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("📋 **Plan de ruta integral para las 5 zonas críticas:**")

st.dataframe(
    top_5_nodos[["nodo_id", "probabilidad_pct", "base_c5", "t_respuesta_min", "hospital", "tiempo_total_min"]],
    column_config={
        "nodo_id": "Nodo Destino",
        "probabilidad_pct": st.column_config.NumberColumn("Riesgo", format="%.1f%%"),
        "base_c5": st.column_config.TextColumn("Unidad C5 (Origen)"),
        "t_respuesta_min": st.column_config.NumberColumn("Arribo (min)", format="%.1f"),
        "hospital": st.column_config.TextColumn("Hospital (Destino)"),
        "tiempo_total_min": st.column_config.NumberColumn("Tiempo Total Operativo", format="%.1f min"),
    },
    hide_index=True,
    use_container_width=True
)