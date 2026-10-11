# Dashboard — Siniestros Viales Iztapalapa

Versión preliminar del dashboard del TT. Cubre hoy la página de inicio y
el panorama histórico; el módulo de pronóstico (GCN+LSTM) está en pausa
en `pronostico_pausado/` y no aparece en la navegación.

## Cómo correrlo

```
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Debe correrse desde DENTRO de esta carpeta (`dashboard/`), porque
`utils/carga_datos.py` ubica `df_hour_clean.csv` de forma relativa (un
nivel arriba, en la raíz del proyecto).

## Estructura

```
dashboard/
├── app.py                          # navegación (st.navigation)
├── app_pages/
│   ├── inicio.py                   # contexto de Iztapalapa, KPIs y metodología
│   └── panorama_historico.py       # EDA: mapa, series de tiempo, rankings
├── pronostico_pausado/
│   └── 2_Pronostico.py             # en pausa: moverlo a app_pages/ para reactivarlo
├── utils/
│   ├── carga_datos.py              # carga y filtrado de df_hour_clean.csv
│   ├── graficas.py                 # funciones de graficación (plotly)
│   ├── mapas.py                    # funciones de mapas (plotly, sin token)
│   └── modelo.py                   # inferencia GCN+LSTM
├── scripts/
│   └── exportar_despues_de_entrenar.py   # plantilla -- ver siguiente sección
├── data/                           # artefactos del modelo (vacío por ahora)
├── .streamlit/config.toml
└── requirements.txt
```

## Pendiente: activar la página de Pronóstico

1. Que termine el entrenamiento (la corrida con GPU).
2. En ESE MISMO notebook y kernel, pegar y correr
   `scripts/exportar_despues_de_entrenar.py` como celda final. Genera en
   `data/`:
   - `modelo_gcn_lstm.pt`
   - `nodos_grafo.csv`
   - `edge_index.npy`
   - `metadatos_modelo.json`
3. Mover `pronostico_pausado/2_Pronostico.py` a `app_pages/pronostico.py`,
   registrarlo en `st.navigation` dentro de `app.py` y recargar el dashboard.

**Por qué tiene que exportarse desde el mismo kernel de entrenamiento y
no reconstruirse por separado**: el orden de los nodos en `nodo_a_idx` y
en `edge_index` depende de esa corrida específica de OSMnx. Reconstruir
el grafo de forma independiente no garantiza el mismo orden, y si no
coincide, los pesos entrenados quedan aplicados a una estructura que ya
no es la misma -- sin que se note como error, solo como un modelo que
predice mal. El detalle completo está en el docstring del script.

## Decisiones ya tomadas (no hace falta volver a discutirlas aquí)

- Filtros de calidad (`coord_valida`, `es_duplicado_c5`) son los mismos
  usados en el análisis de autocorrelación espacial del TT.
- La adyacencia del modelo usa la red vial (OSMnx), no una malla H3 --
  justificado empíricamente con Moran's I
  (`autocorrelacion_espacial_h3_vs_grafo_vial.md`).
- La ventana de entrada (`T = 24` horas) viene de la arquitectura ya
  definida en `modelo_gcn_lstm.ipynb`.

## Limitaciones conocidas de esta versión preliminar

- **Asignación incidente → nodo aproximada**: `construir_ventana_reciente()`
  usa distancia euclidiana (cKDTree) entre el incidente y los nodos del
  grafo, no la topología real de calles que usó
  `ox.distance.nearest_nodes()` durante el entrenamiento. Se documenta
  en el docstring de esa función; si esta ruta llega a producción, vale
  la pena mencionarlo como limitación en el documento del TT.
- **No hay fuente de datos en vivo**: la "hora más reciente" es el final
  del histórico cargado, no el momento actual -- ver la discusión sobre
  la latencia de los datos abiertos de C5. Esta página es una prueba de
  concepto de la tubería de inferencia, no un sistema operando sobre
  datos de hoy.
- Sin integración de una API externa (Waze/Google Maps) por ahora --
  decisión explícita para esta primera versión; queda como trabajo futuro.

## Compartir con el equipo (túneles / port forwarding)

Si expones el dashboard con un túnel (VS Code Dev Tunnels, ngrok, etc.)
y a quien se lo compartes ve la página pero no los datos ni las
gráficas (se queda en "Please wait..." o con espacios vacíos), es un
problema conocido de Streamlit: la conexión WebSocket que transmite el
contenido real no pasa bien por el proxy del túnel. Ya está resuelto en
`.streamlit/config.toml` (`enableCORS = false`,
`enableWebsocketCompression = false`); solo reinicia el dashboard
(`Ctrl+C` y `streamlit run app.py` de nuevo) para que tome el cambio.

Si sigue sin verse, siguiente cosa a revisar: pide a quien lo ve que
abra la consola del navegador (F12 → Console/Network) y busque errores
con "WebSocket" o "Mixed Content" -- eso dice si el problema es del
lado del túnel o de otra cosa.
