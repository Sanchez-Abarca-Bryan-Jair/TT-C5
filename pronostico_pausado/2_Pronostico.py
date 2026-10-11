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
    <h1 style="color: white; font-size: 2.2rem; margin-top: 0; margin-bottom: 10px;">Pronóstico y Asignación Heurística</h1>
    <p style="color: #e0e0e0; font-size: 1.1rem; margin: 0;">Proyección espacial de siniestros viales en Iztapalapa mediante GCN-LSTM</p>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

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

# 1. Cargar Datos e Inferencia
modelo, nodos_df, edge_index, metadatos = cargar_artefactos()
df_incidentes = cargar_incidentes()
T = metadatos.get("T", 24)

matriz_ventana = construir_ventana_reciente(df_incidentes, nodos_df, T, metadatos.get("n_features", 2))
probs = predecir_riesgo(modelo, matriz_ventana, edge_index)
nodos_df["probabilidad_riesgo"] = probs

# ==========================================
# SIDEBAR: AJUSTES DE OPERACIÓN
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
# CÁLCULO LOGÍSTICO PREVIO AL MAPA
# ==========================================
# Preparar Top 20 para ruteo y visualización
df_mostrar = df_critico.sort_values(by="probabilidad_riesgo", ascending=False).head(20).copy()
df_mostrar["probabilidad_pct"] = df_mostrar["probabilidad_riesgo"] * 100

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

arbol_bases = cKDTree(bases_c5[["lat", "lon"]].values)
arbol_hospitales = cKDTree(hospitales_trauma[["lat", "lon"]].values)

# Ruteo para los Nodos Críticos (Top 20)
dist_base, idx_base = arbol_bases.query(df_mostrar[["lat", "lon"]].values)
df_mostrar["base_c5"] = bases_c5.iloc[idx_base]["nombre_base"].values
df_mostrar["t_respuesta_min"] = (dist_base * 111.0 / 30.0) * 60 

dist_hosp, idx_hosp = arbol_hospitales.query(df_mostrar[["lat", "lon"]].values)
df_mostrar["hospital"] = hospitales_trauma.iloc[idx_hosp]["nombre_hospital"].values
df_mostrar["t_traslado_min"] = (dist_hosp * 111.0 / 30.0) * 60 

df_mostrar["tiempo_total_min"] = df_mostrar["t_respuesta_min"] + df_mostrar["t_traslado_min"]
top_5_nodos = df_mostrar.head(5).copy()

# ==========================================
# SECCIÓN: MAPA DE RIESGO
# ==========================================
subtitulo_con_estilo(f"Mapa de Riesgo Espacial (Umbral > {umbral*100:.0f}%)")

opcion_mapa = st.pills(
    "Estilo de visualización",
    options=["Mapa de Calor Térmico", "Top Nodos Operativos", "Hexágonos (3D)"],
    default="Mapa de Calor Térmico"
)

vista_inicial = pdk.ViewState(
    latitude=19.345, 
    longitude=-99.065,
    zoom=11.5,
    pitch=45 if opcion_mapa == "Hexágonos (3D)" else 0,
)

capas = []
if opcion_mapa == "Mapa de Calor Térmico":
    capas.append(pdk.Layer(
        "HeatmapLayer",
        data=df_critico,
        opacity=0.8,
        get_position=["lon", "lat"],
        get_weight="probabilidad_riesgo",
        radius_pixels=40,
    ))
    tooltip_texto = "Zona con riesgo proyectado por GCN-LSTM"

elif opcion_mapa == "Top Nodos Operativos":
    capas.append(pdk.Layer(
        "ScatterplotLayer",
        data=top_5_nodos,
        get_position=["lon", "lat"],
        get_color=[255, 193, 7, 200], # Amarillo alerta
        get_radius=180,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
    ))
    tooltip_texto = "Nodo: {nodo_id}\nRiesgo: {probabilidad_pct}%\nDespacho: {base_c5}\nRecepción: {hospital}"

else: # Hexágonos 3D
    capas.append(pdk.Layer(
        "HexagonLayer",
        data=df_critico,
        get_position=["lon", "lat"],
        radius=150, 
        elevation_scale=50,
        elevation_range=[0, 1000],
        extruded=True,
        get_elevation_weight="probabilidad_riesgo",
        color_range=[
            [254, 217, 118], 
            [253, 141, 60],  
            [227, 26, 28],   
            [128, 0, 38]     
        ],
        pickable=True,
    ))
    tooltip_texto = "Concentración de Riesgo: {elevationValue}"

st.pydeck_chart(pdk.Deck(
    layers=capas,
    initial_view_state=vista_inicial,
    tooltip={"text": tooltip_texto}
))

# ==========================================
# SECCIÓN: TABLAS Y LOGÍSTICA
# ==========================================
col_tabla1, col_tabla2 = st.columns([1, 1.5])

with col_tabla1:
    subtitulo_con_estilo("Top Nodos Críticos")
    st.dataframe(
        df_mostrar[["nodo_id", "probabilidad_pct"]],
        column_config={
            "nodo_id": "ID Nodo",
            "probabilidad_pct": st.column_config.ProgressColumn(
                "Riesgo",
                format="%.1f%%",
                min_value=0, max_value=100,
            ),
        },
        hide_index=True, use_container_width=True, height=250
    )

with col_tabla2:
    subtitulo_con_estilo("Plan Logístico (Doble Tramo)")
    st.markdown("<p style='font-size: 0.9rem; color: gray;'>Heurística de ruta para las 5 zonas críticas principales.</p>", unsafe_allow_html=True)
    
    st.dataframe(
        top_5_nodos[["nodo_id", "base_c5", "hospital", "tiempo_total_min"]],
        column_config={
            "nodo_id": "Nodo",
            "base_c5": st.column_config.TextColumn("Unidad C5"),
            "hospital": st.column_config.TextColumn("Recepción Médica"),
            "tiempo_total_min": st.column_config.NumberColumn("T. Operativo", format="%.1f min"),
        },
        hide_index=True, use_container_width=True, height=250
    )

# Tarjeta de Alerta Máxima
nodo_top_1 = top_5_nodos.iloc[0]
st.error(
    f"**ALERTA PRIORITARIA (Nodo {nodo_top_1['nodo_id']}):** "
    f"Riesgo inminente del {nodo_top_1['probabilidad_pct']:.1f}%. "
    f"Se sugiere pre-despacho desde **{nodo_top_1['base_c5']}** hacia **{nodo_top_1['hospital']}** "
    f"({nodo_top_1['tiempo_total_min']:.1f} min est.).",
    icon="🚨"
)
