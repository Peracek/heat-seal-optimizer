import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
from itertools import product

st.set_page_config(page_title="Heat Seal Parameter Optimizer", layout="wide")

@st.cache_data
def load_data():
    """Load historical data from CSV file."""
    if not os.path.exists('historical_data.csv'):
        st.error("Historical data file 'historical_data.csv' not found!")
        return None
    return pd.read_csv('historical_data.csv')

@st.cache_resource
def load_or_train_model():
    """Load existing model or train new one if it doesn't exist."""
    model_file = 'seal_model.pkl'
    encoder_file = 'encoder.pkl'

    if os.path.exists(model_file) and os.path.exists(encoder_file):
        model = joblib.load(model_file)
        encoder = joblib.load(encoder_file)
        return model, encoder

    # Train new model
    data = load_data()
    if data is None:
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

def main():
    st.title("🔥 Heat Seal Parameter Optimizer")
    st.markdown("Find optimal sealing parameters for your Doypack production")

    # Load model
    model, encoder = load_or_train_model()

    if model is None or encoder is None:
        st.error("Failed to load or train the model. Please ensure historical_data.csv exists.")
        return

    # Load data for UI options
    data = load_data()
    if data is None:
        return

    # Sidebar for inputs
    st.sidebar.header("Production Parameters")

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
    st.sidebar.markdown("**Data Statistics**")
    total_records = len(data)
    pass_rate = (data['Outcome'] == 'Pass').mean() * 100
    st.sidebar.metric("Total Records", total_records)
    st.sidebar.metric("Overall Pass Rate", f"{pass_rate:.1f}%")

if __name__ == "__main__":
    main()