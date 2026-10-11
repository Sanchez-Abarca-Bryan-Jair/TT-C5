"""
Inferencia del modelo GCN+LSTM para la página de Pronóstico.

Este módulo NO reconstruye el grafo vial ni reentrena nada en cada
carga del dashboard: carga artefactos ya exportados desde el notebook
de entrenamiento (modelo_gcn_lstm.ipynb) con la plantilla
scripts/exportar_despues_de_entrenar.py. Mientras esos artefactos no
existan en dashboard/data/, artefactos_disponibles() regresa False y
la página de Pronóstico se limita a explicarlo -- no se simula un
pronóstico con pesos sin entrenar.

Artefactos esperados en dashboard/data/:
    modelo_gcn_lstm.pt      state_dict del modelo entrenado
    nodos_grafo.csv         nodo_id, idx, lat, lon (mismo orden que edge_index)
    edge_index.npy          array (2, E), en índices -- no nodo_id de OSM
    metadatos_modelo.json   T, gcn_hidden, lstm_hidden, n_features usados al entrenar
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

try:
    from torch_geometric.nn import GCNConv
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "La página de Pronóstico necesita torch_geometric instalado en el "
        "mismo entorno donde corre streamlit (pip install torch_geometric)."
    ) from error

# utils/ -> dashboard/ -> data/
RUTA_DATA = Path(__file__).resolve().parents[1] / "data"
RUTA_MODELO = RUTA_DATA / "modelo_gcn_lstm.pt"
RUTA_NODOS = RUTA_DATA / "nodos_grafo.csv"
RUTA_EDGES = RUTA_DATA / "edge_index.npy"
RUTA_METADATOS = RUTA_DATA / "metadatos_modelo.json"


class ModeloGCNLSTM(nn.Module):
    """
    Misma arquitectura definida en modelo_gcn_lstm.ipynb -- se necesita
    tal cual aquí para que load_state_dict() pueda mapear los pesos
    entrenados a esta clase. Si la arquitectura del notebook cambia,
    esta definición debe actualizarse junto con ella.
    """

    def __init__(self, n_features: int, gcn_hidden: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.gcn = GCNConv(n_features, gcn_hidden)
        self.lstm = nn.LSTM(input_size=gcn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.salida = nn.Linear(lstm_hidden, 1)  # logit crudo -- se entrenó con BCEWithLogitsLoss

    def forward(self, ventana_entrada: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """ventana_entrada: (T, N_nodos, n_features)"""
        embeddings_por_hora = []
        for t in range(ventana_entrada.shape[0]):
            x_t = ventana_entrada[t]
            emb_t = torch.relu(self.gcn(x_t, edge_index))
            embeddings_por_hora.append(emb_t)
        secuencia = torch.stack(embeddings_por_hora, dim=1)
        _, (h_final, _) = self.lstm(secuencia)
        return self.salida(h_final[-1])


def artefactos_disponibles() -> bool:
    """True solo si los 4 archivos esperados existen en dashboard/data/."""
    return all(p.exists() for p in [RUTA_MODELO, RUTA_NODOS, RUTA_EDGES, RUTA_METADATOS])


@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def cargar_artefactos():
    """
    Carga el modelo y el grafo una sola vez por sesión de Streamlit.
    Se usa st.cache_resource (no st.cache_data) porque el modelo de
    PyTorch no es un objeto serializable/hasheable como un dataframe.
    """
    with open(RUTA_METADATOS) as f:
        metadatos = json.load(f)

    nodos_df = pd.read_csv(RUTA_NODOS).sort_values("idx").reset_index(drop=True)
    edge_index = torch.tensor(np.load(RUTA_EDGES), dtype=torch.long)

    # El modelo ahora lee dinámicamente si tiene 1 o 2 características desde el JSON
    modelo = ModeloGCNLSTM(
        n_features=metadatos.get("n_features", 1),
        gcn_hidden=metadatos.get("gcn_hidden", 32),
        lstm_hidden=metadatos.get("lstm_hidden", 64),
    )
    modelo.load_state_dict(torch.load(RUTA_MODELO, map_location="cpu"))
    modelo.eval()

    return modelo, nodos_df, edge_index, metadatos


def construir_ventana_reciente(df_incidentes: pd.DataFrame, nodos_df: pd.DataFrame, T: int, n_features: int = 2) -> np.ndarray:
    """
    Construye la matriz (T, N_nodos, n_features) de conteos y tasas.
    Para n_features=2, incluye la tasa histórica normalizada por nodo.
    """
    arbol = cKDTree(nodos_df[["lat", "lon"]].values)

    hora_maxima = df_incidentes["fecha_hora_rango"].max()
    horas_ventana = pd.date_range(end=hora_maxima, periods=T, freq="h")

    # 1. Matriz de conteos dinámicos en la ventana T
    matriz_conteo = np.zeros((T, len(nodos_df)), dtype=np.float32)
    ventana_df = df_incidentes[df_incidentes["fecha_hora_rango"].isin(horas_ventana)]

    if not ventana_df.empty:
        _, indices_nodo = arbol.query(ventana_df[["latitud", "longitud"]].values)
        fila_por_hora = {hora: i for i, hora in enumerate(horas_ventana)}

        for fila_hora, idx_nodo in zip(ventana_df["fecha_hora_rango"].map(fila_por_hora), indices_nodo):
            if pd.notna(fila_hora):
                matriz_conteo[int(fila_hora), idx_nodo] += 1

    # Si el modelo antiguo solo pedía 1 feature, retornamos solo el conteo
    if n_features == 1:
        return matriz_conteo

    # 2. Reconstrucción de la tasa histórica normalizada (Feature 2)
    # Contamos todos los incidentes del historial
    _, indices_totales = arbol.query(df_incidentes[["latitud", "longitud"]].values)
    conteos_historicos = np.bincount(indices_totales, minlength=len(nodos_df))

    # Aplicamos log1p y min-max al igual que en modelo_gcn_lstm.ipynb
    tasa_log = np.log1p(conteos_historicos)
    tasa_norm = tasa_log / (tasa_log.max() + 1e-9)
    tasa_norm = tasa_norm.astype(np.float32)

    # Expandimos a (T, N_nodos) para que cada hora tenga el mismo valor
    ventana_tasa = np.tile(tasa_norm, (T, 1))

    # Apilamos en el último eje para lograr la forma final: (T, N_nodos, 2)
    matriz_final = np.stack([matriz_conteo, ventana_tasa], axis=-1)
    
    return matriz_final


def predecir_riesgo(modelo: "ModeloGCNLSTM", matriz_ventana: np.ndarray, edge_index: torch.Tensor) -> np.ndarray:
    """
    Corre un forward pass y aplica sigmoide.
    """
    entrada = torch.tensor(matriz_ventana, dtype=torch.float32)
    
    # Aseguramos que la entrada tenga la dimensión del canal si solo llegó 2D
    if entrada.ndim == 2:
        entrada = entrada.unsqueeze(-1)
        
    with torch.no_grad():
        logits = modelo(entrada, edge_index)
    return torch.sigmoid(logits).squeeze(-1).numpy()