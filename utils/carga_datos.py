import pandas as pd
import streamlit as st

@st.cache_data(show_spinner="Consultando datos históricos en Neon PostgreSQL...", ttl="1d")
def cargar_incidentes() -> pd.DataFrame:
    # Conexión a Neon usando los secretos de Streamlit
    conn = st.connection("neon_db", type="sql")
    
    # Consulta SQL optimizada: Trae las variables originales + el cruce con el clima y calendario
    query = """
        SELECT 
            s.fecha_creacion, s.hora_creacion, s.dia_semana, s.tipo_incidente_c4,
            s.incidente_c4, s.colonia_catalogo, s.longitud, s.latitud,
            s.anio, s.mes, s.hora, s.es_fin_de_semana, s.franja_horaria,
            s.target, s.fecha_hora, s.fecha_hora_rango,
            -- Nuevas variables exógenas listas para el modelo
            c.tipo_dia_cultural, c.es_quincena, c.es_festivo_oficial,
            cl.temperatura_c, cl.precipitacion_mm
        FROM siniestros_historico s
        LEFT JOIN cat_calendario c ON s.fecha_creacion = c.fecha
        LEFT JOIN cat_clima_historico cl ON s.dt_creacion = cl.dt_hora
        WHERE s.coord_valida = TRUE AND s.es_duplicado_c5 = FALSE;
    """
    
    df = conn.query(query)
    
    # Mantenemos las conversiones por si tu dashboard hace agrupaciones directas con estas columnas
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["fecha_hora_rango"] = pd.to_datetime(df["fecha_hora_rango"])
    
    return df

def filtrar_incidentes(
    df: pd.DataFrame,
    anios: list[int] | None = None,
    dias_semana: list[str] | None = None,
    franjas_horarias: list[str] | None = None,
    tipos_incidente: list[str] | None = None,
    solo_confirmados: bool = True,
) -> pd.DataFrame:
    
    resultado = df.copy()

    if solo_confirmados:
        resultado = resultado[resultado["target"] == 1]
    if anios:
        resultado = resultado[resultado["anio"].isin(anios)]
    if dias_semana:
        resultado = resultado[resultado["dia_semana"].isin(dias_semana)]
    if franjas_horarias:
        resultado = resultado[resultado["franja_horaria"].isin(franjas_horarias)]
    if tipos_incidente:
        resultado = resultado[resultado["incidente_c4"].isin(tipos_incidente)]

    return resultado
