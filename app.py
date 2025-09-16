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

st.set_page_config(page_title="Optimalizátor parametrů tepelného svařování", layout="wide")

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

    # Orders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_code TEXT UNIQUE,
            material_type TEXT,
            print_coverage INTEGER,
            ink_type TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')

    # Attempts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER,
            sealing_temperature_c REAL,
            sealing_pressure_bar REAL,
            dwell_time_s REAL,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    ''')

    # Keep old tables for compatibility
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_type TEXT,
            print_coverage INTEGER,
            ink_type TEXT,
            recommended_temperature_c REAL,
            recommended_pressure_bar REAL,
            recommended_dwell_time_s REAL,
            predicted_success_rate REAL,
            user_feedback TEXT,
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

def save_recommendation_to_db(material_type, print_coverage, ink_type, optimal_params):
    """Save parameter recommendation to database."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO recommendations
        (material_type, print_coverage, ink_type, recommended_temperature_c,
         recommended_pressure_bar, recommended_dwell_time_s, predicted_success_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (str(material_type), int(print_coverage), str(ink_type),
          float(optimal_params['temperature']), float(optimal_params['pressure']),
          float(optimal_params['dwell_time']), float(optimal_params['success_rate'])))
    conn.commit()
    conn.close()

def load_recommendations_from_db():
    """Load recommendations from SQLite database."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    try:
        df = pd.read_sql_query('''
            SELECT id, material_type, print_coverage, ink_type,
                   recommended_temperature_c, recommended_pressure_bar,
                   recommended_dwell_time_s, predicted_success_rate,
                   user_feedback, created_at
            FROM recommendations
            ORDER BY created_at DESC
        ''', conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def update_recommendation_feedback(recommendation_id, feedback):
    """Update recommendation feedback in database."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE recommendations
        SET user_feedback = ?
        WHERE id = ?
    ''', (feedback, recommendation_id))
    conn.commit()
    conn.close()

def create_order(order_code, material_type, print_coverage, ink_type):
    """Create a new order."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO orders (order_code, material_type, print_coverage, ink_type)
            VALUES (?, ?, ?, ?)
        ''', (order_code, material_type, print_coverage, ink_type))
        order_id = cursor.lastrowid
        conn.commit()
        return order_id
    except sqlite3.IntegrityError:
        return None  # Order code already exists
    finally:
        conn.close()

def get_active_order():
    """Get the currently active order."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, order_code, material_type, print_coverage, ink_type, created_at
        FROM orders
        WHERE status = 'active'
        ORDER BY created_at DESC
        LIMIT 1
    ''')
    result = cursor.fetchone()
    conn.close()
    if result:
        return {
            'id': result[0],
            'order_code': result[1],
            'material_type': result[2],
            'print_coverage': result[3],
            'ink_type': result[4],
            'created_at': result[5]
        }
    return None

def complete_order(order_id):
    """Mark an order as completed."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE orders
        SET status = 'completed', completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (order_id,))
    conn.commit()
    conn.close()

def add_attempt(order_id, temperature, pressure, dwell_time, outcome):
    """Add an attempt to an order."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO attempts (order_id, sealing_temperature_c, sealing_pressure_bar, dwell_time_s, outcome)
        VALUES (?, ?, ?, ?, ?)
    ''', (order_id, temperature, pressure, dwell_time, outcome))
    attempt_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return attempt_id

def get_order_attempts(order_id):
    """Get all attempts for an order."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, sealing_temperature_c, sealing_pressure_bar, dwell_time_s, outcome, created_at
        FROM attempts
        WHERE order_id = ?
        ORDER BY created_at ASC
    ''', (order_id,))
    results = cursor.fetchall()
    conn.close()
    return [{
        'id': row[0],
        'temperature': row[1],
        'pressure': row[2],
        'dwell_time': row[3],
        'outcome': row[4],
        'created_at': row[5]
    } for row in results]

@st.cache_data
def load_csv_data():
    """Load historical data from CSV file."""
    if not os.path.exists('historical_data.csv'):
        return pd.DataFrame()
    return pd.read_csv('historical_data.csv')

def load_combined_data():
    """Load and combine CSV, user data, attempts data, and feedback data."""
    csv_data = load_csv_data()
    user_data = load_user_data_from_db()
    attempts_data = load_attempts_data()
    feedback_data = load_feedback_as_training_data()

    # Combine all data sources
    all_data = []
    if not csv_data.empty:
        all_data.append(csv_data)
    if not user_data.empty:
        all_data.append(user_data)
    if not attempts_data.empty:
        # Select only the columns needed for training (same as other sources)
        attempts_subset = attempts_data[['Material_Type', 'Print_Coverage', 'Ink_Type',
                                       'Sealing_Temperature_C', 'Sealing_Pressure_bar',
                                       'Dwell_Time_s', 'Outcome']]
        all_data.append(attempts_subset)
    if not feedback_data.empty:
        all_data.append(feedback_data)

    if not all_data:
        return None

    return pd.concat(all_data, ignore_index=True)

def load_feedback_as_training_data():
    """Convert recommendation feedback into training data."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    try:
        df = pd.read_sql_query('''
            SELECT material_type as Material_Type,
                   print_coverage as Print_Coverage,
                   ink_type as Ink_Type,
                   recommended_temperature_c as Sealing_Temperature_C,
                   recommended_pressure_bar as Sealing_Pressure_bar,
                   recommended_dwell_time_s as Dwell_Time_s,
                   CASE
                       WHEN user_feedback = 'good' THEN 'Pass'
                       WHEN user_feedback = 'bad' THEN 'Fail'
                       ELSE NULL
                   END as Outcome
            FROM recommendations
            WHERE user_feedback IS NOT NULL
        ''', conn)
        # Filter out rows with NULL outcomes
        df = df.dropna(subset=['Outcome'])
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

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
    st.subheader("📝 Přidat nová produkční data")

    # Get existing data for options
    csv_data = load_csv_data()
    if not csv_data.empty:
        material_options = csv_data['Material_Type'].unique().tolist()
        ink_options = csv_data['Ink_Type'].unique().tolist()
    else:
        material_options = [
            'Papír + PET + LDPE',
            'Papír + Al + LDPE',
            'PET + Al + LDPE',
            'BOPP + BOPP + CPP',
            'PET + PETmet + LDPE',
            'BOPP + PETmet + LDPE'
        ]
        ink_options = ['Světlá', 'Tmavá', 'Metalická']

    with st.form("data_entry_form"):
        col1, col2 = st.columns(2)

        with col1:
            material_type = st.selectbox("Typ materiálu", material_options)
            print_coverage = st.slider("Pokrytí tiskem v oblasti svařování (%)", 0, 100, 50)
            ink_type = st.selectbox("Typ barvy v oblasti svařování", ink_options)

        with col2:
            temperature = st.number_input("Teplota svařování (°C)", 100.0, 220.0, 150.0, 1.0)
            pressure = st.number_input("Tlak svařování (bar)", 1.0, 8.0, 4.0, 0.1)
            dwell_time = st.number_input("Doba zdržení (s)", 0.1, 3.0, 1.0, 0.1)

        outcome = st.radio("Výsledek", ["Úspěch", "Neúspěch"], horizontal=True)

        submitted = st.form_submit_button("➕ Přidat datový bod", type="primary")

        if submitted:
            # Validate inputs
            if 100 <= temperature <= 220 and 1.0 <= pressure <= 8.0 and 0.1 <= dwell_time <= 3.0:
                # Translate outcome back to English for consistency with CSV data
                outcome_en = 'Pass' if outcome == 'Úspěch' else 'Fail'
                new_data = {
                    'Material_Type': material_type,
                    'Print_Coverage': print_coverage,
                    'Ink_Type': ink_type,
                    'Sealing_Temperature_C': temperature,
                    'Sealing_Pressure_bar': pressure,
                    'Dwell_Time_s': dwell_time,
                    'Outcome': outcome_en
                }
                save_user_data_to_db(new_data)
                st.session_state.model_needs_retraining = True
                st.success("✅ Datový bod úspěšně přidán! Model bude přetrénován při další predikci.")
                st.rerun()
            else:
                st.error("❌ Neplatné rozsahy parametrů! Zkontrolujte prosím vaše vstupy.")

def render_data_table():
    """Render the data table with current dataset."""
    st.subheader("📊 Aktuální dataset")

    data = load_combined_data()
    if data is not None and not data.empty:
        st.dataframe(data, use_container_width=True)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Přetrénovat model"):
                st.session_state.model_needs_retraining = True
                st.cache_resource.clear()
                st.success("Model bude přetrénován při další predikci!")

        with col2:
            if st.button("📥 Stáhnout data"):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="📄 Stáhnout CSV",
                    data=csv,
                    file_name=f"production_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        # Statistics
        st.subheader("📈 Statistiky datasetu")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Celkem záznamů", len(data))
        with col2:
            pass_rate = (data['Outcome'] == 'Pass').mean() * 100
            st.metric("Úspěšnost", f"{pass_rate:.1f}%")
        with col3:
            st.metric("Typů materiálů", data['Material_Type'].nunique())
        with col4:
            st.metric("Typů barev", data['Ink_Type'].nunique())

    else:
        st.info("Nejsou k dispozici žádná data. Přidejte několik datových bodů nebo se ujistěte, že existuje CSV soubor.")

def optimize_parameters_section(model, encoder, data):
    """Render the parameter optimization section."""
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Vstupní parametry")

        material_options = data['Material_Type'].unique().tolist()
        ink_options = data['Ink_Type'].unique().tolist()

        material_type = st.selectbox(
            "Typ materiálu",
            options=material_options,
            help="Vyberte typ materiálu pro vaši produkční sérii"
        )

        ink_type = st.selectbox(
            "Typ/barva inkoustu v oblasti svařování",
            options=ink_options,
            help="Vyberte dominantní typ nebo barvu inkoustu v oblasti svařování"
        )

        print_coverage = st.slider(
            "Pokrytí tiskem v oblasti svařování (%)",
            min_value=0,
            max_value=100,
            value=50,
            help="Procento povrchu v oblasti svařování pokryté tiskem"
        )

        # Main action button
        optimize_button = st.button("🎯 Najít optimální nastavení", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 Doporučené parametry")

        # Handle button click and display results in the same column
        if optimize_button:
            with st.spinner("Optimalizuji parametry..."):
                optimal_params = find_optimal_parameters(
                    model, encoder, material_type, ink_type, print_coverage
                )

            if optimal_params:
                # Save recommendation to database
                save_recommendation_to_db(material_type, print_coverage, ink_type, optimal_params)

                st.success("✅ Optimální parametry nalezeny!")

                # Display results in metrics
                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "🌡️ Teplota svařování",
                        f"{optimal_params['temperature']:.0f}°C",
                        help="Doporučená teplota svařování"
                    )
                    st.metric(
                        "⚡ Tlak svařování",
                        f"{optimal_params['pressure']:.1f} bar",
                        help="Doporučený tlak svařování"
                    )

                with metric_col2:
                    st.metric(
                        "⏱️ Doba zdržení",
                        f"{optimal_params['dwell_time']:.1f}s",
                        help="Doporučená doba zdržení"
                    )

                # Additional info
                st.info(f"""
                **📋 Shrnutí doporučení:**
                - **Materiál:** {material_type}
                - **Typ barvy v oblasti svařování:** {ink_type}
                - **Pokrytí tiskem v oblasti svařování:** {print_coverage}%

                Tyto parametry jsou optimalizovány na základě historických produkčních dat
                a měly by poskytovat nejvyšší pravděpodobnost úspěšného svaření.
                """)
            else:
                st.error("Nepodařilo se najít optimální parametry. Zkuste prosím jiné vstupy.")
        else:
            st.info("👆 Nastavte parametry a klikněte na tlačítko pro nalezení optimálního nastavení.")

    # Display statistics in sidebar for main page
    if st.sidebar:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Statistiky modelu**")
        total_records = len(data)
        pass_rate = (data['Outcome'] == 'Pass').mean() * 100
        st.sidebar.metric("Celkem záznamů", total_records)
        st.sidebar.metric("Celková úspěšnost", f"{pass_rate:.1f}%")

        # Show data source info
        csv_count = len(load_csv_data()) if not load_csv_data().empty else 0
        user_count = len(load_user_data_from_db()) if not load_user_data_from_db().empty else 0
        attempts_count = len(load_attempts_data()) if not load_attempts_data().empty else 0
        feedback_count = len(load_feedback_as_training_data()) if not load_feedback_as_training_data().empty else 0
        if csv_count > 0 or user_count > 0 or attempts_count > 0 or feedback_count > 0:
            st.sidebar.markdown(f"**Zdroje dat:**")
            if csv_count > 0:
                st.sidebar.markdown(f"• CSV: {csv_count} záznamů")
            if user_count > 0:
                st.sidebar.markdown(f"• Ruční: {user_count} záznamů")
            if attempts_count > 0:
                st.sidebar.markdown(f"• Pokusy: {attempts_count} záznamů")
            if feedback_count > 0:
                st.sidebar.markdown(f"• Zpětná vazba: {feedback_count} záznamů")

def render_recommendation_history():
    """Render the recommendation history with feedback options."""
    st.markdown("---")
    st.subheader("📋 Historie doporučení")

    try:
        recommendations = load_recommendations_from_db()

        if recommendations.empty:
            st.info("Zatím nebyly provedeny žádné doporučení parametrů.")
            return

        st.markdown(f"**Celkem doporučení:** {len(recommendations)}")

        # Display recent recommendations
        for _, rec in recommendations.head(10).iterrows():
            try:
                # Safe conversion of numeric values
                temp = float(rec['recommended_temperature_c']) if pd.notna(rec['recommended_temperature_c']) else 0
                pressure = float(rec['recommended_pressure_bar']) if pd.notna(rec['recommended_pressure_bar']) else 0
                dwell = float(rec['recommended_dwell_time_s']) if pd.notna(rec['recommended_dwell_time_s']) else 0

                with st.expander(f"🕐 {str(rec['created_at'])[:16]} | {rec['material_type']} | {rec['ink_type']} | {rec['print_coverage']}%"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("🌡️ Teplota", f"{temp:.0f}°C")
                        st.metric("⚡ Tlak", f"{pressure:.1f} bar")

                    with col2:
                        st.metric("⏱️ Doba zdržení", f"{dwell:.1f}s")
                        if pd.notna(rec['user_feedback']):
                            feedback_emoji = "👍" if rec['user_feedback'] == "good" else "👎"
                            st.write(f"**Zpětná vazba:** {feedback_emoji}")
                        else:
                            st.write("**Zpětná vazba:** Neohodnoceno")

                    with col3:
                        st.write("**Ohodnotit doporučení po výrobě:**")

                        feedback_key = f"feedback_{rec['id']}"
                        col_good, col_bad = st.columns(2)

                        with col_good:
                            if st.button("👍 Fungovalo", key=f"good_{rec['id']}",
                                        disabled=pd.notna(rec['user_feedback'])):
                                update_recommendation_feedback(rec['id'], "good")
                                st.session_state.model_needs_retraining = True
                                st.success("Zpětná vazba uložena! Model bude přetrénován.")
                                st.rerun()

                        with col_bad:
                            if st.button("👎 Nefungovalo", key=f"bad_{rec['id']}",
                                        disabled=pd.notna(rec['user_feedback'])):
                                update_recommendation_feedback(rec['id'], "bad")
                                st.session_state.model_needs_retraining = True
                                st.error("Zpětná vazba uložena! Model bude přetrénován.")
                                st.rerun()
            except (ValueError, TypeError) as e:
                st.error(f"Chyba při zobrazení doporučení: {e}")
                continue

        if len(recommendations) > 10:
            st.info(f"Zobrazeno posledních 10 doporučení z celkem {len(recommendations)}")

    except Exception as e:
        st.error(f"Chyba při načítání historie doporučení: {e}")
        st.info("Historie doporučení bude dostupná po prvním vygenerování parametrů.")

def render_new_order_form():
    """Render the new order creation form."""
    st.subheader("📋 Nová zakázka")

    # Get material and ink options
    csv_data = load_csv_data()
    if not csv_data.empty:
        material_options = csv_data['Material_Type'].unique().tolist()
        ink_options = csv_data['Ink_Type'].unique().tolist()
    else:
        material_options = [
            'Papír + PET + LDPE',
            'Papír + Al + LDPE',
            'PET + Al + LDPE',
            'BOPP + BOPP + CPP',
            'PET + PETmet + LDPE',
            'BOPP + PETmet + LDPE'
        ]
        ink_options = ['Světlá', 'Tmavá', 'Metalická']

    with st.form("new_order_form"):
        col1, col2 = st.columns(2)

        with col1:
            order_code = st.text_input("Kód zakázky", placeholder="např. Z2024-001")
            material_type = st.selectbox("Typ materiálu", material_options)

        with col2:
            print_coverage = st.slider("Pokrytí tiskem v oblasti svařování (%)", 0, 100, 50)
            ink_type = st.selectbox("Typ barvy v oblasti svařování", ink_options)

        submitted = st.form_submit_button("🚀 Zahájit zakázku", type="primary", use_container_width=True)

        if submitted:
            if order_code.strip():
                order_id = create_order(order_code.strip(), material_type, print_coverage, ink_type)
                if order_id:
                    st.success(f"✅ Zakázka {order_code} byla vytvořena!")
                    st.rerun()
                else:
                    st.error("❌ Zakázka s tímto kódem již existuje!")
            else:
                st.error("❌ Zadejte kód zakázky!")

def render_order_attempts():
    """Render the order attempts interface."""
    active_order = get_active_order()

    if not active_order:
        return

    # Order header
    st.markdown(f"""
    ### 📦 Aktivní zakázka: **{active_order['order_code']}**
    **Materiál:** {active_order['material_type']} | **Barva:** {active_order['ink_type']} | **Pokrytí:** {active_order['print_coverage']}%
    """)

    # Get existing attempts
    attempts = get_order_attempts(active_order['id'])

    # Show existing attempts
    if attempts:
        st.subheader("📊 Historie pokusů")
        for i, attempt in enumerate(attempts, 1):
            outcome_emoji = "✅" if attempt['outcome'] == 'Úspěch' else "❌"
            st.write(f"{outcome_emoji} **Pokus {i}:** {attempt['temperature']}°C, {attempt['pressure']} bar, {attempt['dwell_time']}s - {attempt['outcome']}")

    # Add new attempt form
    st.subheader("🔬 Nový pokus")

    with st.form("attempt_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            temperature = st.number_input("Teplota svařování (°C)", 100.0, 220.0, 150.0, 1.0)
        with col2:
            pressure = st.number_input("Tlak svařování (bar)", 1.0, 8.0, 4.0, 0.1)
        with col3:
            dwell_time = st.number_input("Doba zdržení (s)", 0.1, 3.0, 1.0, 0.1)

        outcome = st.radio("Výsledek pokusu", ["Neúspěch", "Úspěch"], horizontal=True)

        col_submit, col_complete = st.columns([2, 1])

        with col_submit:
            submitted = st.form_submit_button("➕ Přidat pokus", type="primary")

        with col_complete:
            if outcome == "Úspěch":
                complete_submitted = st.form_submit_button("✅ Dokončit zakázku", type="secondary")
            else:
                complete_submitted = False

        if submitted:
            if 100 <= temperature <= 220 and 1.0 <= pressure <= 8.0 and 0.1 <= dwell_time <= 3.0:
                add_attempt(active_order['id'], temperature, pressure, dwell_time, outcome)
                st.success(f"✅ Pokus přidán!")
                st.rerun()
            else:
                st.error("❌ Neplatné rozsahy parametrů!")

        if complete_submitted:
            add_attempt(active_order['id'], temperature, pressure, dwell_time, outcome)
            complete_order(active_order['id'])
            st.success(f"🎉 Zakázka {active_order['order_code']} byla dokončena!")
            st.rerun()

def main_page():
    """Main page for data gathering phase."""
    st.title("🔥 Systém sběru dat tepelného svařování")
    st.markdown("**Fáze 1:** Sběr produkčních dat pro trénování modelu")

    # Check for active order
    active_order = get_active_order()

    if not active_order:
        # No active order - show new order form
        render_new_order_form()
    else:
        # Active order exists - show attempts interface
        render_order_attempts()

        # Option to start new order
        st.markdown("---")
        if st.button("🆕 Nová zakázka", type="secondary"):
            # Complete current order if it has attempts
            attempts = get_order_attempts(active_order['id'])
            if attempts:
                complete_order(active_order['id'])
                st.rerun()
            else:
                st.warning("⚠️ Nejprve přidejte alespoň jeden pokus do aktuální zakázky.")

def load_attempts_data():
    """Load attempts data from database."""
    init_database()
    conn = sqlite3.connect('user_data.db')
    try:
        df = pd.read_sql_query('''
            SELECT o.material_type as Material_Type,
                   o.print_coverage as Print_Coverage,
                   o.ink_type as Ink_Type,
                   a.sealing_temperature_c as Sealing_Temperature_C,
                   a.sealing_pressure_bar as Sealing_Pressure_bar,
                   a.dwell_time_s as Dwell_Time_s,
                   CASE
                       WHEN a.outcome = 'Úspěch' THEN 'Pass'
                       WHEN a.outcome = 'Neúspěch' THEN 'Fail'
                       ELSE a.outcome
                   END as Outcome,
                   o.order_code as Order_Code,
                   a.created_at as Attempt_Date
            FROM attempts a
            JOIN orders o ON a.order_id = o.id
            ORDER BY a.created_at DESC
        ''', conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def data_management_page():
    """Data management page - view only."""
    st.title("📊 Správa produkčních dat")
    st.markdown("Přehled všech produkčních dat z CSV souboru, ručního vstupu a pokusů ze zakázek")

    # Model retraining controls
    st.sidebar.header("⚙️ Nastavení modelu")
    if st.sidebar.button("🔄 Přetrénovat model"):
        st.session_state.model_needs_retraining = True
        st.cache_resource.clear()
        st.sidebar.success("Model bude přetrénován!")

    # Data viewing tabs
    tab1, tab2, tab3 = st.tabs(["📊 Všechna data", "📋 Data ze zakázek", "📄 Historická data"])

    with tab1:
        render_data_table()

    with tab2:
        st.subheader("📋 Data z pokusů ze zakázek")
        attempts_data = load_attempts_data()
        if not attempts_data.empty:
            st.dataframe(attempts_data, use_container_width=True)

            # Statistics
            st.subheader("📈 Statistiky pokusů")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Celkem pokusů", len(attempts_data))
            with col2:
                pass_rate = (attempts_data['Outcome'] == 'Pass').mean() * 100
                st.metric("Úspěšnost pokusů", f"{pass_rate:.1f}%")
            with col3:
                st.metric("Dokončených zakázek", attempts_data['Order_Code'].nunique())
            with col4:
                if not attempts_data.empty:
                    latest_attempt = attempts_data['Attempt_Date'].max()
                    st.metric("Poslední pokus", latest_attempt[:10] if latest_attempt else "N/A")
        else:
            st.info("Zatím nebyly zaznamenány žádné pokusy ze zakázek.")

    with tab3:
        st.subheader("📄 Historická data (CSV)")
        csv_data = load_csv_data()
        if not csv_data.empty:
            st.dataframe(csv_data, use_container_width=True)

            # Statistics
            st.subheader("📈 Statistiky CSV dat")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Celkem záznamů", len(csv_data))
            with col2:
                pass_rate = (csv_data['Outcome'] == 'Pass').mean() * 100
                st.metric("Úspěšnost", f"{pass_rate:.1f}%")
            with col3:
                st.metric("Typů materiálů", csv_data['Material_Type'].nunique())
        else:
            st.info("CSV soubor nebyl nalezen nebo je prázdný.")

def optimization_page():
    """Optimization page - currently disabled, showing data gathering phase message."""
    st.title("🎯 Optimalizace parametrů tepelného svařování")
    st.info("🚧 Fáze optimalizace bude dostupná po dokončení fáze sběru dat.")

    # Show basic stats if data exists
    data = load_combined_data()
    if data is not None and not data.empty:
        model, encoder = load_or_train_model()
        if model is not None and encoder is not None:
            optimize_parameters_section(model, encoder, data)
            render_recommendation_history()
        else:
            st.warning("⚠️ Model nemůže být trénován - nedostatek dat.")
    else:
        st.warning("⚠️ Nejsou k dispozici žádná data pro optimalizaci.")

def main():
    """Main application with page navigation."""
    # Page navigation in sidebar
    st.sidebar.title("🧭 Navigace")
    page = st.sidebar.radio(
        "Vyberte stránku:",
        ["🔥 Sběr dat", "📊 Správa dat", "🎯 Optimalizace parametrů"],
        label_visibility="collapsed"
    )

    # Route to appropriate page
    if page == "🔥 Sběr dat":
        main_page()
    elif page == "📊 Správa dat":
        data_management_page()
    elif page == "🎯 Optimalizace parametrů":
        optimization_page()

if __name__ == "__main__":
    main()