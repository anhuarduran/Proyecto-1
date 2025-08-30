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

    # Eliminar variable "duration of intensive unit stay"
    if "duration of intensive unit stay" in df.columns:
        df = df.drop("duration of intensive unit stay", axis=1)

    return df


# ========================
# USO EN LA APP
# ========================
st.header("📊 Preprocesamiento de los datos")

df_raw = load_data()
if df_raw is not None:
    st.success("✅ Datos cargados correctamente desde GitHub")
    st.write("Dimensiones iniciales:", df_raw.shape)

    st.markdown("""
    ### 🔹 Paso 1: Eliminación de variables irrelevantes  
    Se eliminan:  
    - `BNP` (muchos nulos / ruido).  
    - Identificadores internos: `SNO`, `MRD No.`.  
    - Columna `month year`.  
    """)

    st.markdown("""
    ### 🔹 Paso 2: Conversión de fechas  
    Las variables `D.O.A` (fecha de admisión) y `D.O.D` (fecha de alta) se transforman al formato **datetime**.
    """)

    st.markdown("""
    ### 🔹 Paso 3: Limpieza de variables numéricas  
    Columnas como hemoglobina (HB), glucosa, creatinina, etc. contenían valores como `"EMPTY"`, `"<12"`, o con comas decimales.  
    Estas se normalizan y convierten a numéricas reales.
    """)

    st.markdown("""
    ### 🔹 Paso 4: Transformación de variables categóricas  
    - `GENDER`: M=1, F=0  
    - `RURAL`: R=1, U=0  
    - `TYPE OF ADMISSION-EMERGENCY/OPD`: E=1, O=0  
    - `CHEST INFECTION`: 1/0  
    - `OUTCOME`: convertido a variables dummies  
    """)

    st.markdown("""
    ### 🔹 Paso 5: Eliminación de información no disponible en el ingreso  
    La variable **`duration of intensive unit stay`** se elimina porque no se conoce al momento del ingreso del paciente.
    """)

    # Preprocesar
    df = preprocess_data(df_raw)
    st.success("✅ Datos preprocesados correctamente")
    st.write("Dimensiones después del preprocesamiento:", df.shape)

    # Vista previa
    st.subheader("👀 Vista previa del dataset procesado")
    st.dataframe(df.head(), use_container_width=True)

else:
    st.warning("⚠️ No se pudieron cargar los datos desde GitHub")

# ========================
# EXPLORACIÓN INICIAL (EDA)
# ========================
st.header("🔍 Exploración de Datos (EDA)")

# Eliminar columna que no sirve
if "duration of intensive unit stay" in df.columns:
    df = df.drop("duration of intensive unit stay", axis=1)

# Quitar espacios en los nombres de columnas
df.columns = df.columns.str.strip()

# ========================
# Separación en categóricas y numéricas
# ========================
st.subheader("📑 Separación de variables")

cat_features = [
    'GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD',
    'OUTCOME_DAMA', 'OUTCOME_DISCHARGE', 'OUTCOME_EXPIRY',
    'SMOKING', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD',
    'RAISED CARDIAC ENZYMES', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA',
    'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF',
    'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
    'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
    'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
    'PULMONARY EMBOLISM', 'CHEST INFECTION'
]

num_features = [col for col in df.columns if col not in cat_features and col not in ['D.O.A', 'D.O.D', 'DURATION OF STAY']]

df_numericas = df[num_features]

st.markdown(f"""
- 🔢 Variables **numéricas detectadas**: `{len(num_features)}`  
- 🧾 Variables **categóricas detectadas**: `{len(cat_features)}`
""")

# Mostrar ejemplos
st.write("👀 Vista previa de variables numéricas:")
st.dataframe(df_numericas.head(), use_container_width=True)

# ========================
# Boxplots de las variables numéricas
# ========================
st.subheader("📊 Distribución y posibles outliers (Boxplots)")

st.info("A continuación se muestran diagramas de caja para cada variable numérica con el fin de identificar valores atípicos.")

import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(df_numericas):
    sns.boxplot(x=df[col], ax=axes[i], color="skyblue")
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(axis="x", rotation=45)

# Eliminar ejes vacíos si sobran
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig)

# ========================
# OUTLIERS Y ASIMETRÍA
# ========================
st.header("📌 Análisis de Outliers y Distribuciones")

# ------------------------
# 1. Calcular outliers con IQR
# ------------------------
outliers_list = []
for c in num_features:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = (df[c] < lower) | (df[c] > upper)

    temp = (
        df.loc[mask, [c]]
        .rename(columns={c: "value"})
        .assign(
            variable=c,
            lower_bound=lower,
            upper_bound=upper,
            row_index=lambda x: x.index,
        )
    )

    outliers_list.append(temp)

outliers = pd.concat(outliers_list, ignore_index=True)
resumen = outliers.groupby("variable").size().reset_index(name="n_outliers")
resumen["pct_outliers"] = (resumen["n_outliers"] / len(df)) * 100

st.subheader("📊 Porcentaje de outliers por variable")
st.dataframe(resumen, use_container_width=True)

# ------------------------
# 2. Skewness (asimetría)
# ------------------------
from scipy.stats import skew

asimetria_pandas = df_numericas.skew().sort_values(ascending=False)
altamente_asimetricas = asimetria_pandas[abs(asimetria_pandas) > 2]

st.subheader("📈 Asimetría (Skewness) de variables numéricas")
st.write("Valores positivos indican cola larga a la derecha; negativos, cola a la izquierda.")
st.dataframe(asimetria_pandas, use_container_width=True)

st.warning("Variables con |asimetría| > 2 (fuertemente sesgadas):")
st.write(altamente_asimetricas)

# ------------------------
# 3. Histogramas
# ------------------------
st.subheader("📉 Histogramas de variables numéricas")
fig, ax = plt.subplots(figsize=(12, 8))
df[num_features].hist(bins=50, figsize=(12, 8))
st.pyplot(fig)

# ------------------------
# 4. Interpretación textual
# ------------------------
st.markdown("""
### 📝 Interpretación de distribuciones

**AGE (edad)**  
- Distribución aproximadamente normal con ligera asimetría a la derecha.  
- Mayor concentración entre 50 y 70 años.  

**HB (hemoglobina)**  
- Distribución bastante simétrica.  
- Valores habituales entre 12 y 14 g/dL.  
- Valores extremos (<8 o >18) son poco frecuentes.  

**TLC (total leucocyte count)**  
- Alta asimetría positiva.  
- Mayoría de valores en rangos bajos, pero algunos muy altos generan cola larga.  

**PLATELETS (plaquetas)**  
- Distribución sesgada a la derecha.  
- Mayor densidad entre 200k y 300k, con casos aislados más altos.  

**GLUCOSE (glucosa)**  
- Sesgo positivo pronunciado.  
- Pico en valores bajos, pero con casos >400.  

**UREA**  
- Fuerte asimetría positiva.  
- Mayoría <100, pero con casos extremos elevados.  

**CREATININE (creatinina)**  
- Sesgo positivo fuerte.  
- Valores bajos dominan, pero hay casos altos dispersos.  

**EF (ejection fraction)**  
- Pico importante en 60.  
- Resto distribuido entre 20–40.  
- Genera una distribución no simétrica con concentración en el límite superior.  
""")
