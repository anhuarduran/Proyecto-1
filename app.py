# ============================================================
# Paso 1 — Librerías y configuración Streamlit
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.stats import spearmanr, stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Dependencia opcional para MCA (Análisis de Correspondencias Múltiples)
try:
    import prince
    HAS_PRINCE = True
except ImportError:
    HAS_PRINCE = False

st.set_page_config(page_title="Proyecto ML", layout="wide")
st.title("Proyecto ML - Reducción de Dimensionalidad")

# Verificación de dependencias
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
# 1. Descripción de la base de datos
# ============================================================
st.header("1. Descripción de la base de datos")

st.markdown("""
Este conjunto de datos corresponde a los registros de **14.845 admisiones hospitalarias** (**12.238** pacientes, incluyendo **1.921** con múltiples ingresos) recogidos durante un período de dos años (**1 de abril de 2017** a **31 de marzo de 2019**) en el **Hero DMC Heart Institute**, unidad del **Dayanand Medical College and Hospital** en **Ludhiana, Punjab, India**.

La información incluye:

* **Datos demográficos:** edad, género y procedencia (rural o urbana).
* **Detalles de admisión:** tipo de admisión (emergencia u OPD), fechas de ingreso y alta, duración total de la estancia y **duración en unidad de cuidados intensivos** (columna objetivo en este proyecto).
* **Antecedentes médicos:** tabaquismo, consumo de alcohol, diabetes mellitus (DM), hipertensión (HTN), enfermedad arterial coronaria (CAD), cardiomiopatía previa (CMP), y enfermedad renal crónica (CKD).
* **Parámetros de laboratorio:** hemoglobina (HB), conteo total de leucocitos (TLC), plaquetas, glucosa, urea, creatinina, péptido natriurético cerebral (BNP), enzimas cardíacas elevadas (RCE) y fracción de eyección (EF).
* **Condiciones clínicas y comorbilidades:** más de 28 variables como insuficiencia cardíaca, infarto con elevación del ST (STEMI), embolia pulmonar, shock, infecciones respiratorias, entre otras.
* **Resultado hospitalario:** estado al alta (alta médica o fallecimiento).
""")

st.markdown("""
| Nombre de la variable | Nombre completo | Explicacion breve |
|:---:|:---:|:---:|
| SNO | Serial Number | Número único de registro |
| MRD No. | Admission Number | Número asignado al ingreso |
| D.O.A | Date of Admission | Fecha en que el paciente fue admitido |
| D.O.D | Date of Discharge | Fecha en que el paciente fue dado de alta |
| AGE | AGE | Edad del paciente |
| GENDER | GENDER | Sexo del paciente |
| RURAL | RURAL(R) /Urban(U) | Zona de residencia (rural/urbana) |
| TYPE OF ADMISSION-EMERGENCY/OPD | TYPE OF ADMISSION-EMERGENCY/OPD | Si el ingreso fue por urgencias o consulta externa |
| month year | month year | Mes y año del ingreso |
| DURATION OF STAY | DURATION OF STAY | Días totales de hospitalización |
| duration of intensive unit stay | duration of intensive unit stay | Duración de la estancia en UCI |
| OUTCOME | OUTCOME | Resultado del paciente (alta, fallecimiento, etc.) |
| SMOKING | SMOKING | Historial de consumo de tabaco |
| ALCOHOL | ALCOHOL | Historial de consumo de alcohol |
| DM | Diabetes Mellitus | Diagnóstico de diabetes mellitus |
| HTN | Hypertension | Diagnóstico de hipertensión arterial |
| CAD | Coronary Artery Disease | Diagnóstico de enfermedad coronaria |
| PRIOR CMP | CARDIOMYOPATHY | Historial de miocardiopatía |
| CKD | CHRONIC KIDNEY DISEASE | Diagnóstico de enfermedad renal crónica |
| HB | Haemoglobin | Nivel de hemoglobina en sangre |
| TLC | TOTAL LEUKOCYTES COUNT | Conteo total de leucocitos |
| PLATELETS | PLATELETS | Conteo de plaquetas |
| GLUCOSE | GLUCOSE | Nivel de glucosa en sangre |
| UREA | UREA | Nivel de urea en sangre |
| CREATININE | CREATININE | Nivel de creatinina en sangre |
| BNP | B-TYPE NATRIURETIC PEPTIDE | Péptido relacionado con función cardíaca |
| RAISED CARDIAC ENZYMES | RAISED CARDIAC ENZYMES | Presencia de enzimas cardíacas elevadas |
| EF | Ejection Fraction | Fracción de eyección cardíaca |
| SEVERE ANAEMIA| SEVERE ANAEMIA | Presencia de anemia grave |
| ANAEMIA | ANAEMIA | Presencia de anemia |
| STABLE ANGINA | STABLE ANGINA | Dolor torácico estable por angina |
| ACS | Acute coronary Syndrome | Síndrome coronario agudo |
| STEMI | ST ELEVATION MYOCARDIAL INFARCTION | Infarto agudo de miocardio con elevación del ST |
| ATYPICAL CHEST PAIN | ATYPICAL CHEST PAIN | Dolor torácico no típico |
| HEART FAILURE | HEART FAILURE | Diagnóstico de insuficiencia cardíaca |
| HFREF | HEART FAILURE WITH REDUCED EJECTION FRACTION | Insuficiencia cardíaca con fracción de eyección reducida |
| HFNEF | HEART FAILURE WITH NORMAL EJECTION FRACTION | Insuficiencia cardíaca con fracción de eyección conservada |
| VALVULAR | Valvular Heart Disease | Enfermedad de válvulas cardíacas |
| CHB | Complete Heart Block | Bloqueo cardíaco completo |
| SSS | Sick sinus syndrome | Síndrome de disfunción sinusal |
| AKI | ACUTE KIDNEY INJURY | Lesión renal aguda |
| CVA INFRACT | Cerebrovascular Accident INFRACT | Accidente cerebrovascular isquémico |
| CVA BLEED | Cerebrovascular Accident BLEED | Accidente cerebrovascular hemorrágico |
| AF | Atrial Fibrilation | Fibrilación auricular |
| VT | Ventricular Tachycardia | Taquicardia ventricular |
| PSVT | PAROXYSMAL SUPRA VENTRICULAR TACHYCARDIA | Taquicardia supraventricular paroxística |
| CONGENITAL | Congenital Heart Disease | Enfermedad cardíaca congénita |
| UTI | Urinary tract infection | Infección de vías urinarias |
| NEURO CARDIOGENIC SYNCOPE | NEURO CARDIOGENIC SYNCOPE | Síncope de origen cardiogénico |
| ORTHOSTATIC | ORTHOSTATIC | Hipotensión postural |
| INFECTIVE ENDOCARDITIS | INFECTIVE ENDOCARDITIS | Inflamación de las válvulas cardíacas por infección |
| DVT | Deep venous thrombosis | Trombosis venosa profunda |
| CARDIOGENIC SHOCK | CARDIOGENIC SHOCK | Shock de origen cardíaco |
| SHOCK | SHOCK | Shock por otras causas |
| PULMONARY EMBOLISM | PULMONARY EMBOLISM | Bloqueo de arterias pulmonares por coágulo |
| CHEST INFECTION | CHEST INFECTION | Infección pulmonar |
| DAMA | Discharged Against Medical Advice | Alta médica solicitada por el paciente en contra de la recomendación |
""")

@st.cache_data
def load_and_clean_data():
    url = "https://raw.githubusercontent.com/anhuarduran/Proyecto-1/main/HDHI%20Admission%20data.csv"
    try:
        df_raw = pd.read_csv(url, sep=None, engine="python")
    except Exception as e:
        st.error(f"Error al cargar desde la URL: {e}. Asegúrate de tener el archivo `HDHI Admission data.csv` en la misma carpeta.")
        return None


bd = pd.read_csv('HDHI Admission data.csv')
st.write(bd.head())

# ============================================================
# Sección 2: Pre-procesamiento y Limpieza de datos
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np

# Se asume que la base de datos 'bd' ya ha sido cargada
if 'bd' not in st.session_state:
    st.error("La base de datos 'bd' no ha sido cargada en la aplicación.")
else:
    bd = st.session_state.bd

    st.header("2. Tratamiento de la base de datos")

    @st.cache_data
    def process_data(data):
        st.subheader("2.1 Limpieza y Transformación")
        
        # Mostrar info original
        buffer = st.empty()
        with st.spinner("Analizando información de la base de datos..."):
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                info_output = []
                data.info(buf=lambda s: info_output.append(s))
            info_output = "\n".join(info_output)
            st.code(info_output, language='text')

        # Eliminar la variable BNP
        if 'BNP' in data.columns:
            st.markdown("Se decide eliminar la variable **BNP** dado que tiene más del 50% de valores faltantes.")
            data = data.drop('BNP', axis=1, errors='ignore')
        
        # Eliminar variables innecesarias
        st.markdown("### Eliminar variables innecesarias")
        data = data.drop(['SNO', 'MRD No.', 'month year'], axis=1, errors='ignore')
        
        st.markdown("""
        Teniendo en cuenta que la variable que se refiere a duración en la unidad de cuidados intensivos 
        contiene información que no se tiene cuando un paciente es ingresado al hospital, se decide eliminar con el objetivo de hacer un análisis más realista.
        """)
        
        data = data.drop('duration of intensive unit stay', axis=1, errors='ignore')

        # Normalizar nombres de columnas a minúsculas y snake_case para un manejo más fácil
        data.columns = (
            data.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-_", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace("/", "_", regex=False)
        )
        
        st.markdown("### Transformar variables de fecha y numéricas")
        
        # Transformar las variables de fecha
        if 'd.o.a' in data.columns:
            data['d.o.a'] = pd.to_datetime(data['d.o.a'], format='%m/%d/%Y', errors='coerce')
        if 'd.o.d' in data.columns:
            data['d.o.d'] = pd.to_datetime(data['d.o.d'], format='%m/%d/%Y', errors='coerce')

        # Tratamiento de variables numéricas que están como categóricas
        cols_to_clean = ['hb', 'tlc', 'platelets', 'glucose', 'urea', 'creatinine', 'ef']
        for col in cols_to_clean:
            if col in data.columns:
                data[col] = (
                    data[col]
                    .astype(str)
                    .str.strip()
                    .replace(['EMPTY', 'nan', 'NaN', 'None', ''], np.nan)
                    .str.replace(r'[<>]', '', regex=True)
                    .str.replace(',', '.', regex=False)
                )
                data[col] = pd.to_numeric(data[col], errors='coerce')

        st.markdown("### Mapeo y creación de variables dummy")
        
        # Transforma las variables categóricas a dummies o binarias
        if 'gender' in data.columns:
            data['gender'] = data['gender'].map({'M': 1, 'F': 0})
        if 'rural' in data.columns:
            data['rural'] = data['rural'].map({'R': 1, 'U': 0})
        if 'type_of_admission-emergency/opd' in data.columns:
            data['type_of_admission-emergency/opd'] = data['type_of_admission-emergency/opd'].map({'E': 1, 'O': 0})
        if 'chest_infection' in data.columns:
            data['chest_infection'] = data['chest_infection'].astype(str).map({'1': 1, '0': 0})
        
        if 'outcome' in data.columns:
            data = pd.get_dummies(data, columns=['outcome'], drop_first=False)
        
        # Convierte cualquier columna booleana a int (0 y 1)
        bool_cols = data.select_dtypes(include=bool).columns
        if len(bool_cols) > 0:
            data[bool_cols] = data[bool_cols].astype(int)
        
        return data

    df = process_data(bd.copy())
    st.session_state.df = df
    st.write(df.head())
