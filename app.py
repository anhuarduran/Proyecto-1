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

# ==========================
# RELACIONES BIVARIADAS
# ==========================
st.header("📌 Relaciones Bivariadas")

# -------------------------
# 1. Pairplot: Numéricas + Género
# -------------------------
st.subheader("📊 Dispersión de variables numéricas por género")

st.markdown("""
El siguiente gráfico compara las variables numéricas contra la variable **GÉNERO**:

- **AGE**: No se observan tendencias lineales claras con otras variables.  
- **HB**: Ligera correlación negativa con urea y creatinina; hombres tienden a valores más altos.  
- **TLC**: Muy disperso, sin relación marcada con otras variables.  
- **Plaquetas (PLATELETS)**: No muestra correlaciones fuertes, dispersión amplia en ambos géneros.  
- **Glucose vs Urea/Creatinine**: Casos con glucosa muy alta tienden a mostrar también urea/creatinina altos.  
- **Urea y Creatinine**: Relación positiva clara.  
- **EF (fracción de eyección)**: Concentración en valor 60, con ligera relación inversa con urea/creatinina.  
""")

import seaborn as sns
pair_fig = sns.pairplot(df[num_features + ["GENDER"]], 
                        hue="GENDER", diag_kind="hist", height=2.5)
st.pyplot(pair_fig)


# ==========================
# HOSPITALIZACIONES POR SEXO
# ==========================
st.header("👩‍🦰👨 Hospitalizaciones por Género")

gender_counts = df['GENDER'].value_counts().rename(index={1: 'Masculino', 0: 'Femenino'})

fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis', ax=ax)
ax.set_title("Distribución de Género", fontsize=16)
ax.set_xlabel("Género", fontsize=12)
ax.set_ylabel("Cantidad de Personas", fontsize=12)

for i, value in enumerate(gender_counts.values):
    ax.text(i, value, str(value), ha='center', va='bottom', fontsize=10)

st.pyplot(fig)

st.markdown("""
**Conclusión:**  
La mayor cantidad de pacientes corresponde al **género masculino**.
""")


# ==========================
# HOSPITALIZACIONES POR EDAD
# ==========================
st.header("📅 Hospitalizaciones según Edad")

fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(data=df, x="AGE", bins=20, kde=False, color="blue", ax=ax)
ax.set_title("Distribución de Edades", fontsize=16)
ax.set_xlabel("Edad", fontsize=12)
ax.set_ylabel("Frecuencia", fontsize=12)
st.pyplot(fig)

st.markdown("""
**Conclusión:**

- **Pico de hospitalizaciones**: entre **55 y 63 años**, seguido por 63–68 años.  
- **Asimetría negativa**: la mayor concentración de hospitalizaciones ocurre en personas mayores.  
- **Menor frecuencia**: en edades de **0 a 20 años**, con hospitalizaciones más asociadas a accidentes o condiciones congénitas.  

En general, el **envejecimiento** se relaciona fuertemente con la mayor probabilidad de hospitalización debido a comorbilidades y deterioro natural de la salud.
""")

# ==========================
# EDAD vs DÍAS DE HOSPITALIZACIÓN
# ==========================
st.header("📌 Relación entre Edad y Días de Hospitalización")

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df, x="AGE", y="DURATION OF STAY", alpha=0.6, ax=ax)
sns.regplot(data=df, x="AGE", y="DURATION OF STAY", scatter=False, color="red", ax=ax)

ax.set_title("Relación entre Edad y Días de Hospitalización")
ax.set_xlabel("Edad del paciente")
ax.set_ylabel("Días de hospitalización")
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

# Correlación numérica
corr = df["AGE"].corr(df["DURATION OF STAY"])
st.write(f"**Correlación entre edad y días de hospitalización:** {corr:.4f}")

st.markdown("""
📊 **Interpretación:**  
- Existe una correlación **positiva muy débil** (~0.106).  
- Aunque los pacientes de mayor edad tienden a permanecer un poco más en el hospital, la relación **no es fuerte**.  
- La gran dispersión de los puntos confirma que **otros factores clínicos y comorbilidades** influyen más en la duración de la hospitalización.  
""")


# ==========================
# 3. División Train/Test
# ==========================
st.header("⚙️ División en Conjuntos de Entrenamiento y Prueba")

# Definimos variables predictoras y objetivo
X = df[num_features + cat_features]
y = df["DURATION OF STAY"]

# División
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

st.write(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} registros")
st.write(f"Tamaño del conjunto de prueba: {X_test.shape[0]} registros")

st.markdown("""
La variable objetivo es **DURATION OF STAY** (días de hospitalización).  
Su predicción es de gran valor para la **planificación clínica y operativa**, permitiendo optimizar:  
- Disponibilidad de camas 🛏️  
- Asignación de personal 👩‍⚕️👨‍⚕️  
- Gestión de recursos hospitalarios ⚕️  
""")


# ==========================
# 2.2 Preprocesamiento
# ==========================
st.header("🔧 Preprocesamiento de Datos")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Pipeline para numéricas
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", RobustScaler())
])

# Pipeline para categóricas
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# Aplicar
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

st.success("✅ Preprocesamiento aplicado correctamente: imputación y escalado realizados.")

# ==========================
# 4. Selección de características: PCA
# ==========================
st.header("🧩 Selección de Características - PCA")

# 1. Extraer numéricas procesadas
num_indices = [i for i, col in enumerate(X_train.columns) if col in num_features]
cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

X_train_numericas = pd.DataFrame(
    X_train_processed[:, num_indices],
    columns=num_features
)
X_test_numericas = pd.DataFrame(
    X_test_processed[:, num_indices],
    columns=num_features
)

# También necesitamos las categóricas para MCA
X_train_categoricas = pd.DataFrame(
    X_train_processed[:, cat_indices],
    columns=cat_features
)
X_test_categoricas = pd.DataFrame(
    X_test_processed[:, cat_indices],
    columns=cat_features
)

# 2. PCA (90% var. explicada)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.90, random_state=42)
Xn_train_pca = pca.fit_transform(X_train_numericas)
Xn_test_pca  = pca.transform(X_test_numericas)

pca_names = [f"PCA{i+1}" for i in range(Xn_train_pca.shape[1])]
Xn_train_pca = pd.DataFrame(Xn_train_pca, columns=pca_names, index=X_train.index)
Xn_test_pca  = pd.DataFrame(Xn_test_pca,  columns=pca_names, index=X_test.index)

st.success(f"PCA generó **{len(pca_names)} componentes**, con una varianza acumulada explicada de **{pca.explained_variance_ratio_.sum():.3f}**.")

# 3. Gráfico de varianza explicada
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(range(1, len(var_exp)+1), var_exp, alpha=0.6, label="Varianza explicada por componente")
ax.step(range(1, len(cum_var_exp)+1), cum_var_exp, where="mid", color="red", label="Varianza acumulada")
ax.axhline(y=0.9, color="green", linestyle="--", label="90%")
ax.set_xlabel("Componentes principales")
ax.set_ylabel("Proporción de varianza explicada")
ax.set_title("PCA - Varianza explicada y acumulada")
ax.set_xticks(range(1, len(var_exp)+1))
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ==========================
# 4.1. PCA: Heatmap de Loadings
# ==========================
st.subheader("📌 PCA - Heatmap de Loadings")

loadings = pd.DataFrame(pca.components_, columns=num_features, index=pca_names)

fig3, ax3 = plt.subplots(figsize=(12, 8))
sns.heatmap(loadings.T, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax3)
ax3.set_title("Mapa de calor de los loadings del PCA")
ax3.set_xlabel("Componentes principales")
ax3.set_ylabel("Variables originales")
st.pyplot(fig3)

# ==========================
# 4.2. MCA: Análisis de Correspondencias Múltiples
# ==========================
st.subheader("📌 MCA - Variables Categóricas")

import prince
mca = prince.MCA(n_components=10, random_state=42)
mca = mca.fit(X_train_categoricas)

Xc_train_mca = mca.transform(X_train_categoricas)
Xc_test_mca  = mca.transform(X_test_categoricas)

mca_names = [f"MCA{i+1}" for i in range(Xc_train_mca.shape[1])]
Xc_train_mca = pd.DataFrame(Xc_train_mca, columns=mca_names, index=X_train.index)
Xc_test_mca  = pd.DataFrame(Xc_test_mca,  columns=mca_names, index=X_test.index)

# Eigenvalues resumen
ev = mca.eigenvalues_summary.copy()
ev["% of variance"] = ev["% of variance"].replace("%", "", regex=True)
ev["% of variance"] = ev["% of variance"].str.replace(",", ".", regex=False)
ev["% of variance"] = pd.to_numeric(ev["% of variance"], errors="coerce")

var_exp_mca = ev["% of variance"].values / 100
cum_var_exp_mca = np.cumsum(var_exp_mca)
componentes = np.arange(1, len(var_exp_mca) + 1)

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.bar(componentes, var_exp_mca, alpha=0.7, label="Varianza explicada")
ax4.plot(componentes, cum_var_exp_mca, marker="o", color="red", label="Varianza acumulada")
ax4.set_xticks(componentes)
ax4.set_xlabel("Componentes MCA")
ax4.set_ylabel("Proporción de varianza")
ax4.set_title("Scree plot - MCA")
ax4.legend()
ax4.grid(alpha=0.3)
st.pyplot(fig4)

# ==========================
# 4.3. Heatmap MCA - Loadings
# ==========================
st.subheader("📌 MCA - Heatmap de Categorías vs Componentes")

coords = mca.column_coordinates(X_train_categoricas)
coords.index = coords.index.astype(str)

threshold = 0.2
filtered_data = coords.loc[:, coords.abs().max() > threshold]

fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.heatmap(filtered_data, cmap="coolwarm", center=0, ax=ax5)
ax5.set_title("Heatmap de loadings - MCA")
ax5.set_xlabel("Componentes")
ax5.set_ylabel("Categorías")
st.pyplot(fig5)

# ==========================
# 4.4. Scatterplot MCA
# ==========================
st.subheader("📌 MCA - Scatterplot (2 primeras componentes)")

row_coords = mca.row_coordinates(X_train_categoricas)

fig6, ax6 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=row_coords[0], y=row_coords[1], alpha=0.6, ax=ax6)

explained_variance_ratio = mca.eigenvalues_
ax6.set_xlabel(f"Componente 1 ({explained_variance_ratio[0]*100:.2f}%)")
ax6.set_ylabel(f"Componente 2 ({explained_variance_ratio[1]*100:.2f}%)")
ax6.set_title("MCA - Scatterplot de las dos primeras componentes")
ax6.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig6)

# ==========================
# 4.5. Dataset reducido
# ==========================
X_train_reduced = pd.concat([Xn_train_pca, Xc_train_mca], axis=1)
X_test_reduced  = pd.concat([Xn_test_pca,  Xc_test_mca],  axis=1)

st.success(f"✅ Shape train reducido: {X_train_reduced.shape}, Shape test reducido: {X_test_reduced.shape}")


# ============================
# 4) PRE-PRUNING con RandomizedSearchCV (solo TRAIN)
# ============================
st.header("🌳 Árbol de Decisión - Pre-Pruning")

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

pipe_pre = Pipeline(steps=[
    ('model', DecisionTreeRegressor(random_state=42))
])

param_dist_pre = {
    'model__max_depth': np.arange(2, 20),
    'model__min_samples_split': np.arange(10, 50),
    'model__min_samples_leaf': np.arange(5, 20),
    'model__max_features': [None, 'sqrt', 'log2']
}

search_pre = RandomizedSearchCV(
    estimator=pipe_pre,
    param_distributions=param_dist_pre,
    n_iter=60,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    refit=True
)

with st.spinner("🔎 Entrenando modelo con RandomizedSearchCV..."):
    search_pre.fit(X_train_reduced, y_train)

# Mostrar mejores hiperparámetros
st.subheader("📌 Resultados del Pre-Pruning")
st.write("**Mejores hiperparámetros encontrados:**")
st.json(search_pre.best_params_)

st.write(f"🏆 Mejor RMSE (CV): {-search_pre.best_score_:.4f}")

# ============================
# Evaluación en test
# ============================
y_pred_pre = search_pre.predict(X_test_reduced)
rmse_test_pre = np.sqrt(mean_squared_error(y_test, y_pred_pre))
r2_test_pre = r2_score(y_test, y_pred_pre)
rmse_rel_test_pre = rmse_test_pre / y_test.mean()

st.success(f"""
**Evaluación en Test:**
- 🔎 RMSE: {rmse_test_pre:.3f}  
- 🔎 R²: {r2_test_pre:.3f}  
- 🔎 RMSE relativo: {rmse_rel_test_pre:.3f}  
""")

# ============================
# 5) Curva de aprendizaje
# ============================
st.subheader("📊 Curva de Aprendizaje")

train_sizes, train_scores, val_scores = learning_curve(
    estimator=search_pre.best_estimator_,
    X=X_train_reduced,
    y=y_train,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
)

train_rmse = -np.mean(train_scores, axis=1)
val_rmse = -np.mean(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')
ax.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validación')
ax.set_title('Curva de Aprendizaje - Árbol de Decisión con Pre-Pruning')
ax.set_xlabel('Tamaño del conjunto de entrenamiento')
ax.set_ylabel('RMSE')
ax.legend(loc='best')
ax.grid(True)

st.pyplot(fig)

# ============================
# 📌 Conclusiones Pre-pruning
# ============================
st.header("📌 Conclusiones Pre-pruning")

st.markdown("""
El proceso de **pre-pruning** se aplicó con el objetivo de optimizar el modelo de árbol de decisión antes de su entrenamiento completo, evitando la creación de un árbol excesivamente complejo.

- La búsqueda de los mejores hiperparámetros identificó una configuración óptima con `max_depth = 3` y `min_samples_leaf = 7`.
- La curva de aprendizaje mostró **sobreajuste**:  
  - Error de entrenamiento ≈ 0.  
  - Error de validación ≈ 4.75 (no mejora con más datos).  
- En test: RMSE = **4.6892**, consistente con el CV (≈4.7832).  
- Sin embargo, la diferencia con el error de entrenamiento confirma que el modelo **no generaliza bien**.

👉 Para mejorar: usar más regularización o modelos más robustos como **Random Forest**.
""")

# ============================
# 5) POST-PRUNING (árbol grande -> path alphas -> CV en TRAIN)
# ============================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

# === 0) Cast a float32 si aplica (acelera y reduce memoria)
X_train_np = np.asarray(X_train_reduced, dtype=np.float32)
y_train_np = np.asarray(y_train)  # target puede quedar en float64
X_test_np  = np.asarray(X_test_reduced,  dtype=np.float32)
y_test_np  = np.asarray(y_test)

# === 1) Ruta de poda en un SUBSET
rng = np.random.RandomState(42)
n_sub = min(len(X_train_np), 15000)
idx = rng.choice(len(X_train_np), n_sub, replace=False)

tree_full_sub = DecisionTreeRegressor(random_state=42)
tree_full_sub.fit(X_train_np[idx], y_train_np[idx])

path = tree_full_sub.cost_complexity_pruning_path(X_train_np[idx], y_train_np[idx])
alphas_full = np.unique(np.round(path.ccp_alphas, 10))

# === 2) Muestrea ~40 alphas representativos (cuantiles)
if len(alphas_full) > 40:
    quantiles = np.linspace(0, 1, 40)
    ccp_alphas = np.quantile(alphas_full, quantiles)
else:
    ccp_alphas = alphas_full

# === 3) CV paralela y árbol con límites de complejidad (más rápido)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

def cv_rmse_for_alpha(a):
    model = DecisionTreeRegressor(
        random_state=42,
        ccp_alpha=a,
        max_depth=6,
        min_samples_leaf=11
    )
    scores = cross_val_score(
        model, X_train_np, y_train_np,
        scoring='neg_root_mean_squared_error',
        cv=kf, n_jobs=-1
    )
    return float((-scores).mean())

rmse_list = [cv_rmse_for_alpha(a) for a in ccp_alphas]
best_idx   = int(np.argmin(rmse_list))
best_alpha = float(ccp_alphas[best_idx])

st.subheader("🌳 Post-pruning")
st.write(f"Mejor `ccp_alpha`: {best_alpha:.6f} | RMSE CV = {rmse_list[best_idx]:.4f}")

# === 4) Entrena SOLO el árbol final
tree_pruned = DecisionTreeRegressor(
    random_state=42,
    ccp_alpha=best_alpha,
    max_depth=20,
    min_samples_leaf=5
)
tree_pruned.fit(X_train_np, y_train_np)

n_leaves = tree_pruned.get_n_leaves()
n_nodes  = tree_pruned.tree_.node_count
st.write(f"🌿 Hojas: {n_leaves} | 🔗 Nodos: {n_nodes}")

# --- Curva de Aprendizaje con el árbol podado ---
train_sizes, train_scores, val_scores = learning_curve(
    tree_pruned,
    X_train_np, y_train_np,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

# Convertir MSE negativo a RMSE
train_rmse = np.sqrt(-train_scores.mean(axis=1))
val_rmse   = np.sqrt(-val_scores.mean(axis=1))

# --- Gráfico ---
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(train_sizes, train_rmse, 'o-', color='blue', label='Entrenamiento')
ax.plot(train_sizes, val_rmse, 'o-', color='orange', label='Validación')
ax.set_title("Curva de Aprendizaje - Árbol Podado")
ax.set_xlabel("Tamaño del conjunto de entrenamiento")
ax.set_ylabel("RMSE")
ax.legend(loc='best')
ax.grid(True)
st.pyplot(fig)

# --- Evaluación final ---
y_pred_post = tree_pruned.predict(X_test_np)
rmse_test_post = np.sqrt(mean_squared_error(y_test_np, y_pred_post))
r2_test_post = r2_score(y_test_np, y_pred_post)
rmse_rel_test_post = rmse_test_post / y_test_np.mean()

st.subheader("📊 Evaluación Final Post-pruning")
st.write(f"🔎 RMSE: {rmse_test_post:.3f}")
st.write(f"🔎 R²: {r2_test_post:.3f}")
st.write(f"🔎 RMSE relativo (test): {rmse_rel_test_post:.3f}")

# === Conclusiones Post-pruning ===
st.subheader("📝 Conclusiones Post-pruning")
st.write("""
El modelo logró encontrar el **ccp_alpha óptimo = 0.034431**, lo que resultó en un árbol con 
**70 hojas y 139 nodos**, un tamaño significativamente menor que un árbol sin podar.  

📉 El **RMSE de validación cruzada** fue de **4.821**, valor comparable al de otros modelos, 
pero el análisis de la curva de aprendizaje muestra un comportamiento problemático:

- El **error de entrenamiento** comienza bajo pero aumenta a medida que se agregan más datos.  
- El **error de validación** comienza alto y se estabiliza.  
- Las dos curvas **nunca convergen**, lo que indica un claro **subajuste**.  

👉 En conclusión, a pesar de la poda, el árbol **no logra aprender adecuadamente las relaciones en los datos**, 
incluso cuando se amplía el conjunto de entrenamiento. Se recomienda explorar modelos más robustos 
(Random Forest, Gradient Boosting) para mejorar la capacidad de generalización.
""")

