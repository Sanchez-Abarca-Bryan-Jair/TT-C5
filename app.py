"""
Herramienta prospectiva de siniestros viales en Iztapalapa -- Dashboard.
Punto de entrada: define la navegación. Las vistas viven en app_pages/.

Ejecutar (desde dentro de esta carpeta):
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Siniestros viales Iztapalapa",
    page_icon=":material/traffic:",
    layout="wide",
)

pagina = st.navigation(
    [
        st.Page(
            "app_pages/inicio.py",
            title="Inicio",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "app_pages/panorama_historico.py",
            title="Panorama histórico",
            icon=":material/bar_chart:",
        ),
        st.Page(
            "pronostico_pausado/2_Pronostico.py",
            title="Pronóstico de Riesgo",
            icon=":material/online_prediction:",
        ),
    ],
    position="top",
)

pagina.run()
