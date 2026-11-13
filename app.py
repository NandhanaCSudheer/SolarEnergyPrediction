import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN APPLICATION
       ===================================================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #f7fbff 0%,
            #eef8f3 50%,
            #fffaf0 100%
        );
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* =====================================================
       FORCE MAIN PAGE TEXT TO DARK COLOR
       ===================================================== */

    .stApp,
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp div {
        color: #173b32;
    }

    /* Main headings */
    .stApp h1 {
        color: #0b3d2e !important;
        font-weight: 800 !important;
    }

    .stApp h2 {
        color: #0d4a38 !important;
        font-weight: 750 !important;
    }

    .stApp h3 {
        color: #145c45 !important;
        font-weight: 700 !important;
    }

    /* Paragraph text */
    .stApp p {
        color: #40574f !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0c3b2c,
            #14543e
        );
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: white !important;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #dce8e2;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.06);
    }

    div[data-testid="stMetric"] label {
        color: #557268 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0b3d2e !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #245c49 !important;
    }


    /* =====================================================
       INPUT LABELS
       ===================================================== */

    .stNumberInput label {
        color: #173b32 !important;
        font-weight: 650 !important;
    }


    /* =====================================================
       INPUT BOXES
       ===================================================== */

    div[data-baseweb="input"] {
        background-color: white !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        color: #173b32 !important;
        background-color: white !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       PREDICTION BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 13px;
        font-size: 18px;
        font-weight: 750;

        background: linear-gradient(
            90deg,
            #f4b400,
            #ffc928
        ) !important;

        color: #3a2a00 !important;

        border: none !important;

        box-shadow: 0 5px 15px rgba(244, 180, 0, 0.20);
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #e9a900,
            #f7bd16
        ) !important;

        color: #241b00 !important;

        transform: translateY(-1px);
    }


    /* =====================================================
       SUCCESS MESSAGE
       ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stAlert"] p {
        color: inherit !important;
    }


    /* =====================================================
       EXPANDER
       ===================================================== */

    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.85);
        border: 1px solid #dce8e2;
        border-radius: 14px;
    }

    div[data-testid="stExpander"] p {
        color: #40574f !important;
    }


    /* =====================================================
       DIVIDERS
       ===================================================== */

    hr {
        border-color: #d6e4de !important;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-text {
        color: #557268 !important;
        text-align: center;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_model.pkl")


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("☀️ Solar Predictor")

    st.divider()

    st.subheader("📌 About the App")

    st.write(
        """
        This application predicts **solar AC power**
        using environmental conditions and time-based
        features.
        """
    )

    st.divider()

    st.subheader("🤖 Machine Learning")

    st.write("🌳 Random Forest Regressor")
    st.write("📈 Regression Problem")
    st.write("⚡ Target: AC Power")

    st.divider()

    st.subheader("📊 Input Features")

    st.write("🌡️ Ambient Temperature")
    st.write("🔥 Module Temperature")
    st.write("☀️ Irradiation")
    st.write("🕐 Hour")
    st.write("⏱️ Minute")
    st.write("📅 Day")
    st.write("🗓️ Month")

    st.divider()

    st.caption("Solar Energy Prediction Project")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("☀️ Solar Power Predictor")

st.subheader(
    "Machine Learning Based Solar AC Power Prediction"
)

st.write(
    """
    Enter the environmental and time-related conditions below
    to estimate the expected solar AC power output.
    """
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("🤖 Model Performance")

st.write(
    "Performance of the Random Forest model on the chronological test dataset."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🌳 Model",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="R² Score",
        value="0.9792"
    )

with col3:
    st.metric(
        label="RMSE",
        value="52.7437"
    )

with col4:
    st.metric(
        label="MAE",
        value="17.5101"
    )


st.divider()


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

st.header("🌤️ Environmental Conditions")

st.write(
    "Enter the environmental conditions used by the trained model."
)

col1, col2, col3 = st.columns(3)

with col1:

    ambient_temperature = st.number_input(
        "🌡️ Ambient Temperature (°C)",
        min_value=-50.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

with col2:

    module_temperature = st.number_input(
        "🔥 Module Temperature (°C)",
        min_value=-50.0,
        max_value=100.0,
        value=35.0,
        step=0.1
    )

with col3:

    irradiation = st.number_input(
        "☀️ Irradiation",
        min_value=0.0,
        value=0.8,
        step=0.01
    )


# ============================================================
# TIME INFORMATION
# ============================================================

st.header("🕐 Time Information")

st.write(
    "Enter the date and time conditions for the prediction."
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )

with col2:

    minute = st.number_input(
        "Minute",
        min_value=0,
        max_value=59,
        value=0,
        step=1
    )

with col3:

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=15,
        step=1
    )

with col4:

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=5,
        step=1
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

st.header("🔮 Generate Prediction")

predict_button = st.button(
    "☀️ Predict Solar Power",
    use_container_width=True,
    type="primary"
)


if predict_button:

    input_data = pd.DataFrame(
        [{
            "AMBIENT_TEMPERATURE": ambient_temperature,
            "MODULE_TEMPERATURE": module_temperature,
            "IRRADIATION": irradiation,
            "HOUR": hour,
            "MINUTE": minute,
            "DAY": day,
            "MONTH": month
        }]
    )

    prediction = model.predict(input_data)[0]

    st.session_state["prediction"] = prediction


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    st.success(
        "✅ Prediction generated successfully!"
    )

    st.metric(
        label="⚡ Predicted Solar AC Power",
        value=f"{prediction:.2f}"
    )

    st.info(
        "The prediction was generated using the trained "
        "Random Forest regression model."
    )

    if prediction < 100:

        st.warning(
            "🌥️ The predicted solar output is relatively low."
        )

    elif prediction < 500:

        st.info(
            "⛅ The predicted solar output is moderate."
        )

    else:

        st.success(
            "☀️ The predicted solar output is relatively high."
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header("📘 Project Information")

with st.expander("🔍 View Project Details"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📌 Project")

        st.write(
            "**Project:** Solar Energy Generation Prediction"
        )

        st.write(
            "**Problem Type:** Regression"
        )

        st.write(
            "**Target Variable:** AC_POWER"
        )

        st.write(
            "**Dataset:** Solar generation and weather data"
        )

    with col2:

        st.subheader("🧠 Machine Learning")

        st.write(
            "**Baseline Model:** Linear Regression"
        )

        st.write(
            "**Final Model:** Random Forest Regressor"
        )

        st.write(
            "**Important Feature:** Irradiation"
        )

    st.divider()

    st.subheader("🎯 Objective")

    st.write(
        """
        The objective of this project is to predict solar AC power
        using environmental conditions and time-based features.

        The model uses:

        - Ambient Temperature
        - Module Temperature
        - Irradiation
        - Hour
        - Minute
        - Day
        - Month
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        ☀️ Solar Energy Generation Prediction
        &nbsp; | &nbsp;
        Machine Learning Regression Project
    </div>
    """,
    unsafe_allow_html=True
)