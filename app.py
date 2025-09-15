import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
from itertools import product
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Heat Seal Parameter Optimizer", layout="wide")

# Initialize session state for data management
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'csv'
if 'user_data' not in st.session_state:
    st.session_state.user_data = []
if 'model_needs_retraining' not in st.session_state:
    st.session_state.model_needs_retraining = False

def init_database():
    """Initialize SQLite database for user data."""
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS production_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_type TEXT,
            print_coverage INTEGER,
            ink_type TEXT,
            sealing_temperature_c REAL,
            sealing_pressure_bar REAL,
            dwell_time_s REAL,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def load_user_data_from_db():
    """Load user data from SQLite database."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    try:
        df = pd.read_sql_query('''
            SELECT material_type as Material_Type,
                   print_coverage as Print_Coverage,
                   ink_type as Ink_Type,
                   sealing_temperature_c as Sealing_Temperature_C,
                   sealing_pressure_bar as Sealing_Pressure_bar,
                   dwell_time_s as Dwell_Time_s,
                   outcome as Outcome
            FROM production_data
        ''', conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def save_user_data_to_db(data):
    """Save user data to SQLite database."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO production_data
        (material_type, print_coverage, ink_type, sealing_temperature_c, sealing_pressure_bar, dwell_time_s, outcome)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (data['Material_Type'], data['Print_Coverage'], data['Ink_Type'],
          data['Sealing_Temperature_C'], data['Sealing_Pressure_bar'],
          data['Dwell_Time_s'], data['Outcome']))
    conn.commit()
    conn.close()

@st.cache_data
def load_csv_data():
    """Load historical data from CSV file."""
    if not os.path.exists('historical_data.csv'):
        return pd.DataFrame()
    return pd.read_csv('historical_data.csv')

def load_combined_data():
    """Load and combine CSV and user data."""
    csv_data = load_csv_data()
    user_data = load_user_data_from_db()

    if csv_data.empty and user_data.empty:
        return None

    if csv_data.empty:
        return user_data
    elif user_data.empty:
        return csv_data
    else:
        return pd.concat([csv_data, user_data], ignore_index=True)

@st.cache_resource
def load_or_train_model():
    """Load existing model or train new one if it doesn't exist."""
    model_file = 'seal_model.pkl'
    encoder_file = 'encoder.pkl'

    if (os.path.exists(model_file) and os.path.exists(encoder_file) and
        not st.session_state.model_needs_retraining):
        model = joblib.load(model_file)
        encoder = joblib.load(encoder_file)
        return model, encoder

    # Train new model
    data = load_combined_data()
    if data is None or data.empty:
        return None, None

    # Prepare features and target
    categorical_features = ['Material_Type', 'Ink_Type']
    numerical_features = ['Print_Coverage', 'Sealing_Temperature_C', 'Sealing_Pressure_bar', 'Dwell_Time_s']

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(data[categorical_features])

    # Combine features
    X = np.hstack([
        categorical_encoded,
        data[numerical_features].values
    ])

    # Target variable (convert Pass/Fail to 1/0)
    y = (data['Outcome'] == 'Pass').astype(int)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model and encoder
    joblib.dump(model, model_file)
    joblib.dump(encoder, encoder_file)

    # Reset retraining flag
    st.session_state.model_needs_retraining = False

    return model, encoder

def find_optimal_parameters(model, encoder, material_type, ink_type, print_coverage):
    """Find optimal sealing parameters using grid search."""
    # Define parameter ranges
    temperature_range = np.arange(120, 181, 5)  # 120-180°C, step 5
    pressure_range = np.arange(3.0, 6.1, 0.2)   # 3-6 bar, step 0.2
    dwell_time_range = np.arange(0.5, 2.1, 0.1) # 0.5-2.0 s, step 0.1

    best_params = None
    best_probability = 0

    # Grid search
    for temp, pressure, dwell_time in product(temperature_range, pressure_range, dwell_time_range):
        # Prepare input for prediction
        categorical_data = pd.DataFrame({
            'Material_Type': [material_type],
            'Ink_Type': [ink_type]
        })

        categorical_encoded = encoder.transform(categorical_data)

        features = np.hstack([
            categorical_encoded,
            [[print_coverage, temp, pressure, dwell_time]]
        ])

        # Get probability of success
        prob = model.predict_proba(features)[0][1]  # Probability of 'Pass'

        if prob > best_probability:
            best_probability = prob
            best_params = {
                'temperature': temp,
                'pressure': pressure,
                'dwell_time': dwell_time,
                'success_rate': prob
            }

    return best_params

def render_data_entry_form():
    """Render the data entry form."""
    st.subheader("📝 Add New Production Data")

    # Get existing data for options
    csv_data = load_csv_data()
    if not csv_data.empty:
        material_options = csv_data['Material_Type'].unique().tolist()
        ink_options = csv_data['Ink_Type'].unique().tolist()
    else:
        material_options = ['PET/ALU/PE', 'PE-Coated Paper']
        ink_options = ['Light', 'Dark', 'Metallic']

    with st.form("data_entry_form"):
        col1, col2 = st.columns(2)

        with col1:
            material_type = st.selectbox("Material Type", material_options)
            print_coverage = st.slider("Print Coverage (%)", 0, 100, 50)
            ink_type = st.selectbox("Ink Type", ink_options)

        with col2:
            temperature = st.number_input("Sealing Temperature (°C)", 100.0, 220.0, 150.0, 1.0)
            pressure = st.number_input("Sealing Pressure (bar)", 1.0, 8.0, 4.0, 0.1)
            dwell_time = st.number_input("Dwell Time (s)", 0.1, 3.0, 1.0, 0.1)

        outcome = st.radio("Outcome", ["Pass", "Fail"], horizontal=True)

        submitted = st.form_submit_button("➕ Add Data Point", type="primary")

        if submitted:
            # Validate inputs
            if 100 <= temperature <= 220 and 1.0 <= pressure <= 8.0 and 0.1 <= dwell_time <= 3.0:
                new_data = {
                    'Material_Type': material_type,
                    'Print_Coverage': print_coverage,
                    'Ink_Type': ink_type,
                    'Sealing_Temperature_C': temperature,
                    'Sealing_Pressure_bar': pressure,
                    'Dwell_Time_s': dwell_time,
                    'Outcome': outcome
                }
                save_user_data_to_db(new_data)
                st.session_state.model_needs_retraining = True
                st.success("✅ Data point added successfully! Model will retrain on next prediction.")
                st.rerun()
            else:
                st.error("❌ Invalid parameter ranges! Please check your inputs.")

def render_data_table():
    """Render the data table with current dataset."""
    st.subheader("📊 Current Dataset")

    data = load_combined_data()
    if data is not None and not data.empty:
        st.dataframe(data, use_container_width=True)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Retrain Model"):
                st.session_state.model_needs_retraining = True
                st.cache_resource.clear()
                st.success("Model will retrain on next prediction!")

        with col2:
            if st.button("📥 Download Data"):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"production_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        # Statistics
        st.subheader("📈 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            pass_rate = (data['Outcome'] == 'Pass').mean() * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        with col3:
            st.metric("Material Types", data['Material_Type'].nunique())
        with col4:
            st.metric("Ink Types", data['Ink_Type'].nunique())

    else:
        st.info("No data available. Add some data points or ensure CSV file exists.")

def optimize_parameters_section(model, encoder, data):
    """Render the parameter optimization section."""
    # Sidebar for inputs
    st.sidebar.header("🎯 Production Parameters")

    material_options = data['Material_Type'].unique().tolist()
    ink_options = data['Ink_Type'].unique().tolist()

    material_type = st.sidebar.selectbox(
        "Material Type",
        options=material_options,
        help="Select the material type for your production run"
    )

    ink_type = st.sidebar.selectbox(
        "Ink Type/Color",
        options=ink_options,
        help="Select the dominant ink type or color"
    )

    print_coverage = st.sidebar.slider(
        "Print Coverage (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="Percentage of package surface covered with print"
    )

    # Main action button
    if st.sidebar.button("🎯 Find Optimal Settings", type="primary"):
        with st.spinner("Optimizing parameters..."):
            optimal_params = find_optimal_parameters(
                model, encoder, material_type, ink_type, print_coverage
            )

        if optimal_params:
            st.success("✅ Optimal parameters found!")

            # Display results in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🌡️ Temperature",
                    f"{optimal_params['temperature']:.0f}°C",
                    help="Recommended sealing temperature"
                )

            with col2:
                st.metric(
                    "⚡ Pressure",
                    f"{optimal_params['pressure']:.1f} bar",
                    help="Recommended sealing pressure"
                )

            with col3:
                st.metric(
                    "⏱️ Dwell Time",
                    f"{optimal_params['dwell_time']:.1f}s",
                    help="Recommended dwell time"
                )

            with col4:
                st.metric(
                    "🎯 Success Rate",
                    f"{optimal_params['success_rate']*100:.1f}%",
                    help="Predicted success probability"
                )

            # Additional info
            st.info(f"""
            **Recommendation Summary:**
            - Material: {material_type}
            - Ink Type: {ink_type}
            - Print Coverage: {print_coverage}%

            These parameters are optimized based on historical production data
            and should provide the highest probability of successful sealing.
            """)
        else:
            st.error("Could not find optimal parameters. Please try different inputs.")

    # Display some statistics about the data
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Data Statistics**")
    total_records = len(data)
    pass_rate = (data['Outcome'] == 'Pass').mean() * 100
    st.sidebar.metric("Total Records", total_records)
    st.sidebar.metric("Overall Pass Rate", f"{pass_rate:.1f}%")

    # Show data source info
    csv_count = len(load_csv_data()) if not load_csv_data().empty else 0
    user_count = len(load_user_data_from_db()) if not load_user_data_from_db().empty else 0
    if csv_count > 0 and user_count > 0:
        st.sidebar.markdown(f"**Data Sources:**")
        st.sidebar.markdown(f"• CSV: {csv_count} records")
        st.sidebar.markdown(f"• Manual: {user_count} records")

def main():
    st.title("🔥 Heat Seal Parameter Optimizer")
    st.markdown("Find optimal sealing parameters for your Doypack production")

    # Data Management Section
    st.sidebar.header("📊 Data Management")

    data_source = st.sidebar.radio(
        "Data Source",
        ["CSV File Only", "Manual Input + CSV", "Manual Input Only"],
        help="Choose how to manage your production data"
    )

    st.session_state.data_source = data_source

    # Load model based on data source
    model, encoder = load_or_train_model()

    if model is None or encoder is None:
        st.error("Failed to load or train the model. Please add data or ensure historical_data.csv exists.")
        if data_source != "CSV File Only":
            st.info("👇 You can add data manually below to get started!")
        return

    # Load data for UI options
    data = load_combined_data()
    if data is None or data.empty:
        if data_source != "CSV File Only":
            st.info("👇 Add some production data to get started!")
        else:
            return

    # Main content area
    if data_source in ["Manual Input + CSV", "Manual Input Only"]:
        tab1, tab2, tab3 = st.tabs(["🎯 Optimize Parameters", "📝 Add Data", "📊 View Data"])

        with tab2:
            render_data_entry_form()

        with tab3:
            render_data_table()

        with tab1:
            optimize_parameters_section(model, encoder, data)
    else:
        optimize_parameters_section(model, encoder, data)

if __name__ == "__main__":
    main()