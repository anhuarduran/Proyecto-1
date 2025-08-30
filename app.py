# ============================================================
# Paso 1 — Librerías y configuración Streamlit
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, LabelEncoder
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import prince
from sklearn.cluster import KMeans

# Configuración de página
st.set_page_config(page_title="Proyecto ML - Dimensionalidad", layout="wide")
st.title("Proyecto ML - Reducción de Dimensionalidad")

# Estado de dependencias
try:
    import prince
    HAS_PRINCE = True
except ImportError:
    HAS_PRINCE = False

col1, col2 = st.columns(2)
with col1:
    st.success("✅ Librerías base y de ML disponibles")
with col2:
    if HAS_PRINCE:
        st.success("✅ `prince` instalado (MCA habilitado)")
    else:
        st.warning(
            "ℹ️ `prince` no está instalado. "
            "Para habilitar MCA, agrega `prince` a tu `requirements.txt` "
            "o instala localmente con `pip install prince`."
        )

# ============================================================
# 1. Cargar y Pre-procesar datos (función con cache)
# ============================================================
@st.cache_data
def load_and_clean_data():
    url = "https://raw.githubusercontent.com/Juansebastianrde/Reduccion-de-dimensionalidad/main/HDHI%20Admission%20data.csv"
    try:
        df_raw = pd.read_csv(url, sep=None, engine="python")
    except Exception as e:
        st.error(f"Error al cargar desde la URL: {e}.")
        return None

    # Normalizar nombres de columnas a minúsculas y snake_case para un manejo más fácil
    df_raw.columns = (
        df_raw.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-_", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    df = df_raw.drop(columns=["sno", "mrd_no", "month_year", "bnp", "duration_of_intensive_unit_stay"], errors="ignore").copy()
    
    if "doa" in df.columns:
        df["doa"] = pd.to_datetime(df["doa"], format="%m/%d/%Y", errors="coerce")
    if "dod" in df.columns:
        df["dod"] = pd.to_datetime(df["dod"], format="%m/%d/%Y", errors="coerce")
    
    cols_to_clean = ["hb", "tlc", "platelets", "glucose", "urea", "creatinine", "ef"]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip()
                       .replace(["EMPTY", "nan", "NaN", "None", ""], np.nan)
                       .str.replace(r"[<>]", "", regex=True)
                       .str.replace(",", ".", regex=False))
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.upper().map({"M": 1, "F": 0})
    if "rural" in df.columns:
        df["rural"] = df["rural"].astype(str).str.strip().str.upper().map({"R": 1, "U": 0})
    if "type_of_admission-emergency_opd" in df.columns:
        df["type_of_admission-emergency_opd"] = (
            df["type_of_admission-emergency_opd"].astype(str).str.strip()
            .str.upper().map({"E": 1, "O": 0})
        )
    if "chest_infection" in df.columns:
        df["chest_infection"] = pd.to_numeric(df["chest_infection"], errors="coerce").astype("Int64")
    
    if "outcome" in df.columns:
        df = pd.get_dummies(df, columns=["outcome"], drop_first=False, dtype=int)
    
    bool_cols = df.select_dtypes(include=bool).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Identificación de variables categóricas y numéricas
    cat_features_raw = [
        'gender', 'rural', 'type_of_admission-emergency_opd',
        'outcome_dama', 'outcome_discharge', 'outcome_expiry',
        'smoking', 'alcohol', 'dm', 'htn', 'cad', 'prior_cmp', 'ckd',
        'raised_cardiac_enzymes', 'severe_anaemia', 'anaemia', 'stable_angina',
        'acs', 'stemi', 'atypical_chest_pain', 'heart_failure', 'hfref', 'hfnef',
        'valvular', 'chb', 'sss', 'aki', 'cva_infract', 'cva_bleed', 'af', 'vt', 'psvt',
        'congenital', 'uti', 'neuro_cardiogenic_syncope', 'orthostatic',
        'infective_endocarditis', 'dvt', 'cardiogenic_shock', 'shock',
        'pulmonary_embolism', 'chest_infection'
    ]
    
    cat_features = [c for c in cat_features_raw if c in df.columns]
    exclude = [c for c in ['doa', 'dod', 'duration_of_stay'] if c in df.columns]
    num_features = [c for c in df.columns if c not in cat_features + exclude]

    st.session_state["df"] = df
    st.session_state["cat_features"] = cat_features
    st.session_state["num_features"] = num_features

    return df

st.header("1. Carga y pre-procesamiento de datos")
if "df" not in st.session_state:
    with st.spinner("Cargando y procesando los datos..."):
        df = load_and_clean_data()
    if df is not None:
        st.success(f"Datos cargados: {df.shape}")
        st.dataframe(df.head(), use_container_width=True)
else:
    df = st.session_state["df"]
    st.success("Datos ya cargados.")
    st.dataframe(df.head(), use_container_width=True)

# ============================================================
# 2. Análisis Exploratorio de Datos (EDA)
# ============================================================
st.header("2. Análisis Exploratorio de Datos (EDA)")

with st.expander("Resumen de columnas (nulos y dtypes)", expanded=False):
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "unique": df.nunique(dropna=True)
    })
    summary["%nulls"] = (summary["nulls"] / len(df) * 100).round(2)
    summary = summary[["dtype", "non_null", "nulls", "%nulls", "unique"]]
    st.dataframe(summary, use_container_width=True)
