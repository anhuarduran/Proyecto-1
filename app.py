# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

import prince  # MCA, CA, FAMD


# ========================
# CONFIGURACIÓN APP
# ========================
st.set_page_config(page_title="🏥 Hospital Admissions", layout="wide")
st.title("🏥 Hospital Admissions Analysis")
st.markdown("Aplicación convertida desde Google Colab → Streamlit ✔️")


# ========================
# CARGA DE DATOS
# ========================
@st.cache_data
def load_data():
    """
    Carga los datos directamente desde la URL de GitHub.
    Si hay un error en la carga, retorna None.
    """
    url = "https://raw.githubusercontent.com/Juansebastianrde/Reduccion-de-dimensionalidad/main/HDHI%20Admission%20data.csv"
    try:
        df = pd.read_csv(url, sep=",", engine="python")  # 👈 Aseguramos que es coma
        return df
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {e}")
        return None


df = load_data()

if df is not None:
    st.success("✅ Datos cargados correctamente desde GitHub")

    st.subheader("👀 Vista previa")
    st.write("Dimensiones del dataset:", df.shape)
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📋 Tipos de variables")
    st.write(df.dtypes)

else:
    st.warning("⚠️ No se pudieron cargar los datos desde GitHub")

# ========================
# PREPROCESAMIENTO DE LA BASE
# ========================
@st.cache_data
def preprocess_data(bd: pd.DataFrame) -> pd.DataFrame:
    """Limpieza y transformación del dataset hospitalario"""

    # Eliminar variable BNP (demasiados nulos / ruido)
    if "BNP" in bd.columns:
        bd = bd.drop("BNP", axis=1)

    # Eliminar variables innecesarias
    cols_to_drop = [c for c in ["SNO", "MRD No.", "month year"] if c in bd.columns]
    df = bd.drop(columns=cols_to_drop)

    # Fechas a datetime
    if "D.O.A" in df.columns:
        df["D.O.A"] = pd.to_datetime(df["D.O.A"], format="%m/%d/%Y", errors="coerce")
    if "D.O.D" in df.columns:
        df["D.O.D"] = pd.to_datetime(df["D.O.D"], format="%m/%d/%Y", errors="coerce")

    # Variables numéricas que vienen como texto
    cols_to_clean = ["HB", "TLC", "PLATELETS", "GLUCOSE", "UREA", "CREATININE", "EF"]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace(["EMPTY", "nan", "NaN", "None", ""], np.nan)
                .str.replace(r"[<>]", "", regex=True)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Variables categóricas → binarias
    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})
    if "RURAL" in df.columns:
        df["RURAL"] = df["RURAL"].map({"R": 1, "U": 0})
    if "TYPE OF ADMISSION-EMERGENCY/OPD" in df.columns:
        df["TYPE OF ADMISSION-EMERGENCY/OPD"] = df["TYPE OF ADMISSION-EMERGENCY/OPD"].map(
            {"E": 1, "O": 0}
        )
    if "CHEST INFECTION" in df.columns:
        df["CHEST INFECTION"] = df["CHEST INFECTION"].map({"1": 1, "0": 0})

    # Dummies para outcome
    if "OUTCOME" in df.columns:
        df = pd.get_dummies(df, columns=["OUTCOME"], drop_first=False)

    # Booleanas a int
    bool_cols = df.select_dtypes(include=bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Eliminar variable "duration of intensive unit stay" (no disponible al ingreso)
    if "duration of intensive unit stay" in df.columns:
        df = df.drop("duration of intensive unit stay", axis=1)

    return df


# ========================
# USO EN LA APP
# ========================
df_raw = load_data()
if df_raw is not None:
    st.success("✅ Datos cargados")
    st.write("Dimensiones iniciales:", df_raw.shape)

    # Preprocesar
    df = preprocess_data(df_raw)
    st.success("✅ Datos preprocesados")
    st.write("Dimensiones después del preprocesamiento:", df.shape)

    # Vista previa
    st.subheader("👀 Vista previa (después de limpieza)")
    st.dataframe(df.head(), use_container_width=True)

else:
    st.warning("⚠️ No se pudieron cargar los datos")

