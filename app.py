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


import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# Paso 1: Carga y almacenamiento de datos
# ============================================================
@st.cache_data
def load_data():
    """
    Carga los datos directamente desde la URL de GitHub.
    Si hay un error en la carga, retorna None.
    """
    url = "https://raw.githubusercontent.com/Juansebastianrde/Reduccion-de-dimensionalidad/main/HDHI%20Admission%20data.csv"
    try:
        df = pd.read_csv(url, sep=None, engine="python")
        return df
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {e}")
        return None

if 'bd' not in st.session_state:
    st.session_state.bd = load_data()

bd = st.session_state.bd

# ============================================================
# Paso 2: Mostrar información del DataFrame
# ============================================================
st.header("Información de la base de datos")

if bd is not None:
    st.success("✅ ¡Base de datos cargada correctamente!")
    
    # Captura y muestra el resultado de bd.info()
    info_buffer = []
    bd.info(buf=lambda s: info_buffer.append(s))
    info_output = "\n".join(info_buffer)
    
    st.code(info_output, language='text')
else:
    st.warning("La base de datos no se pudo cargar. Revisa la URL y tu conexión.")
