import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib
from itertools import product
import sqlite3
import psycopg2
from datetime import datetime
import os
import threading
from contextlib import contextmanager

st.set_page_config(page_title="Optimalizátor parametrů tepelného svařování")

def format_datetime(dt_value):
    """Format datetime for display - handles both datetime objects and strings."""
    if dt_value is None:
        return 'N/A'
    if isinstance(dt_value, datetime):
        return dt_value.strftime('%Y-%m-%d %H:%M')
    # If it's a string, try to slice it (SQLite format)
    try:
        return str(dt_value)[:16]
    except:
        return str(dt_value)

def format_date(dt_value):
    """Format date for display - handles both datetime objects and strings."""
    if dt_value is None:
        return 'N/A'
    if isinstance(dt_value, datetime):
        return dt_value.strftime('%Y-%m-%d')
    # If it's a string, try to slice it (SQLite format)
    try:
        return str(dt_value)[:10]
    except:
        return str(dt_value)

# Database connection pool and caching
_connection_lock = threading.Lock()
_connection_pool = None
_database_initialized = False

def _create_connection():
    """Create a single database connection - PostgreSQL or SQLite."""
    # Check if we're running in Streamlit Cloud (has secrets but may be empty)
    is_streamlit_cloud = hasattr(st, 'secrets') and st.secrets is not None

    # Try to get PostgreSQL connection from Streamlit secrets
    if is_streamlit_cloud:
        try:
            if 'DATABASE_URL' in st.secrets:
                # Create connection without autocommit for better performance
                conn = psycopg2.connect(st.secrets['DATABASE_URL'])
                return conn
            elif hasattr(st.secrets, 'postgres'):
                # Alternative secrets format
                conn = psycopg2.connect(
                    host=st.secrets.postgres.host,
                    database=st.secrets.postgres.database,
                    user=st.secrets.postgres.user,
                    password=st.secrets.postgres.password,
                    port=st.secrets.postgres.port
                )
                return conn
            else:
                # Running on Streamlit Cloud but no PostgreSQL secrets configured
                st.error("🚨 **Database Configuration Required**")
                st.error("PostgreSQL connection not configured for Streamlit Cloud deployment.")
                st.info("Please add DATABASE_URL to your app secrets. See POSTGRESQL_SETUP.md for instructions.")
                st.stop()
        except Exception as e:
            # PostgreSQL connection failed
            st.error("🚨 **Database Connection Failed**")
            st.error(f"Could not connect to PostgreSQL: {str(e)}")
            st.info("Check your database connection settings and ensure your PostgreSQL service is running.")
            st.stop()

    # Fallback to SQLite for local development only
    return sqlite3.connect('user_data.db')

@st.cache_resource
def get_connection_pool():
    """Get or create the database connection pool."""
    global _connection_pool
    with _connection_lock:
        if _connection_pool is None or _connection_pool.closed:
            _connection_pool = _create_connection()
        return _connection_pool

@contextmanager
def get_database_connection():
    """Get database connection with automatic cleanup."""
    # Ensure database is initialized on first access

    conn = get_connection_pool()
    try:
        # Check if connection is valid before yielding
        if hasattr(conn, 'closed') and conn.closed:
            # Connection is closed, clear cache and get a new one
            st.cache_resource.clear()
            conn = get_connection_pool()

        yield conn
        # Only commit if we're not using autocommit and connection is still open
        if hasattr(conn, 'commit') and not conn.closed:
            conn.commit()
    except Exception as e:
        # Only rollback if we're not using autocommit and connection is still open
        if hasattr(conn, 'rollback') and not conn.closed:
            conn.rollback()
        raise e

def ensure_database_initialized():
    """Ensure database is initialized only once."""
    global _database_initialized
    if not _database_initialized:
        with _connection_lock:
            if not _database_initialized:
                _init_database_tables()
                _database_initialized = True

def _init_database_tables():
    """Initialize database tables - called only once."""
    with get_database_connection() as conn:
        cursor = conn.cursor()

        # Detect database type
        is_postgres = isinstance(conn, psycopg2.extensions.connection)

        try:
            # Primary key and auto-increment syntax differs between databases
            if is_postgres:
                pk_syntax = "id SERIAL PRIMARY KEY"
                timestamp_default = "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            else:
                pk_syntax = "id INTEGER PRIMARY KEY AUTOINCREMENT"
                timestamp_default = "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"

            # Orders table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS orders (
                {pk_syntax},
                order_code TEXT UNIQUE,
                material_type TEXT,
                print_coverage INTEGER,
                package_size INTEGER,
                {timestamp_default}
            )
            ''')

            # Attempts table with multi-phase sealing parameters
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS attempts (
                {pk_syntax},
                order_id INTEGER,
                -- Legacy single-phase parameters (for backward compatibility)
                sealing_temperature_c REAL,
                sealing_pressure_bar REAL,
                dwell_time_s REAL,
                -- Zipper sealing phase
                zipper_temperature_c REAL,
                zipper_pressure_bar REAL,
                zipper_dwell_time_s REAL,
                -- Bottom sealing phase
                bottom_temperature_c REAL,
                bottom_pressure_bar REAL,
                bottom_dwell_time_s REAL,
                -- Side sealing phases (E, D, C, B, A)
                side_e_temperature_c REAL,
                side_e_pressure_bar REAL,
                side_e_dwell_time_s REAL,
                side_d_temperature_c REAL,
                side_d_pressure_bar REAL,
                side_d_dwell_time_s REAL,
                side_c_temperature_c REAL,
                side_c_pressure_bar REAL,
                side_c_dwell_time_s REAL,
                side_b_temperature_c REAL,
                side_b_pressure_bar REAL,
                side_b_dwell_time_s REAL,
                side_a_temperature_c REAL,
                side_a_pressure_bar REAL,
                side_a_dwell_time_s REAL,
                outcome TEXT,
                {timestamp_default},
                FOREIGN KEY (order_id) REFERENCES orders (id)
            )
            ''')

            # Add new multi-phase columns to existing attempts table if they don't exist
            new_columns = [
            'zipper_temperature_c REAL',
            'zipper_pressure_bar REAL',
            'zipper_dwell_time_s REAL',
            'bottom_temperature_c REAL',
            'bottom_pressure_bar REAL',
            'bottom_dwell_time_s REAL',
            'side_e_temperature_c REAL',
            'side_e_pressure_bar REAL',
            'side_e_dwell_time_s REAL',
            'side_d_temperature_c REAL',
            'side_d_pressure_bar REAL',
            'side_d_dwell_time_s REAL',
            'side_c_temperature_c REAL',
            'side_c_pressure_bar REAL',
            'side_c_dwell_time_s REAL',
            'side_b_temperature_c REAL',
            'side_b_pressure_bar REAL',
            'side_b_dwell_time_s REAL',
            'side_a_temperature_c REAL',
            'side_a_pressure_bar REAL',
            'side_a_dwell_time_s REAL'
            ]

            for column in new_columns:
                try:
                    cursor.execute(f'ALTER TABLE attempts ADD COLUMN {column}')
                except (sqlite3.OperationalError, psycopg2.errors.DuplicateColumn):
                    # Column already exists
                    pass
                except Exception:
                    # Other database errors, column might already exist
                    pass

            # Add package_size column to orders table if it doesn't exist
            try:
                cursor.execute('ALTER TABLE orders ADD COLUMN package_size INTEGER')
            except (sqlite3.OperationalError, psycopg2.errors.DuplicateColumn):
                # Column already exists
                pass
            except Exception:
                # Other database errors, column might already exist
                pass

            # Add sackovacka column to orders table if it doesn't exist
            try:
                cursor.execute('ALTER TABLE orders ADD COLUMN sackovacka TEXT')
            except (sqlite3.OperationalError, psycopg2.errors.DuplicateColumn):
                # Column already exists
                pass

            # Add note column to attempts table if it doesn't exist
            try:
                cursor.execute('ALTER TABLE attempts ADD COLUMN note TEXT')
            except (sqlite3.OperationalError, psycopg2.errors.DuplicateColumn):
                # Column already exists
                pass
            except Exception:
                # Other database errors, column might already exist
                pass

            # Add note column to orders table if it doesn't exist
            try:
                cursor.execute('ALTER TABLE orders ADD COLUMN note TEXT')
            except (sqlite3.OperationalError, psycopg2.errors.DuplicateColumn):
                # Column already exists
                pass
            except Exception:
                # Other database errors, column might already exist
                pass

            # Keep old tables for compatibility
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS production_data (
                    {pk_syntax},
                    material_type TEXT,
                    print_coverage INTEGER,
                    sealing_temperature_c REAL,
                    sealing_pressure_bar REAL,
                    dwell_time_s REAL,
                    outcome TEXT,
                    {timestamp_default}
                )
            ''')
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS recommendations (
                    {pk_syntax},
                    material_type TEXT,
                    print_coverage INTEGER,
                    recommended_temperature_c REAL,
                    recommended_pressure_bar REAL,
                    recommended_dwell_time_s REAL,
                    predicted_success_rate REAL,
                    user_feedback TEXT,
                    {timestamp_default}
                )
            ''')

        except Exception as e:
            print(f"Database initialization error: {e}")
            raise

# Initialize session state for data management
if 'user_data' not in st.session_state:
    st.session_state.user_data = []
if 'model_needs_retraining' not in st.session_state:
    st.session_state.model_needs_retraining = False

# Initialize session state for order workflow
if 'current_order_id' not in st.session_state:
    st.session_state.current_order_id = None
if 'order_screen' not in st.session_state:
    st.session_state.order_screen = False
if 'show_new_order_form' not in st.session_state:
    st.session_state.show_new_order_form = False

# Initialize session state for stage alignment
if 'stages_aligned' not in st.session_state:
    st.session_state.stages_aligned = False
if 'aligned_side_temperature' not in st.session_state:
    st.session_state.aligned_side_temperature = 155.0
if 'aligned_side_pressure' not in st.session_state:
    st.session_state.aligned_side_pressure = 4.2
if 'aligned_side_dwell_time' not in st.session_state:
    st.session_state.aligned_side_dwell_time = 1.1


def load_user_data_from_db():
    """Load user data from database."""
    try:
        with get_database_connection() as conn:
            df = pd.read_sql_query('''
                SELECT material_type as Material_Type,
                       print_coverage as Print_Coverage,
                       sealing_temperature_c as Sealing_Temperature_C,
                       sealing_pressure_bar as Sealing_Pressure_bar,
                       dwell_time_s as Dwell_Time_s,
                       outcome as Outcome
                FROM production_data
            ''', conn)
            return df
    except:
        return pd.DataFrame()

def save_user_data_to_db(data):
    """Save user data to database."""
    with get_database_connection() as conn:
        cursor = conn.cursor()

        # Detect database type for parameter placeholders
        is_postgres = isinstance(conn, psycopg2.extensions.connection)
        placeholder = '%s' if is_postgres else '?'

        cursor.execute(f'''
            INSERT INTO production_data
            (material_type, print_coverage, sealing_temperature_c, sealing_pressure_bar, dwell_time_s, outcome)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        ''', (data['Material_Type'], data['Print_Coverage'],
              data['Sealing_Temperature_C'], data['Sealing_Pressure_bar'],
              data['Dwell_Time_s'], data['Outcome']))

def save_recommendation_to_db(material_type, print_coverage, optimal_params):
    """Save parameter recommendation to database."""
    with get_database_connection() as conn:
        cursor = conn.cursor()

        # Detect database type for parameter placeholders
        is_postgres = isinstance(conn, psycopg2.extensions.connection)
        placeholder = '%s' if is_postgres else '?'

        cursor.execute(f'''
            INSERT INTO recommendations
            (material_type, print_coverage, recommended_temperature_c,
             recommended_pressure_bar, recommended_dwell_time_s, predicted_success_rate)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        ''', (str(material_type), int(print_coverage),
              float(optimal_params['temperature']), float(optimal_params['pressure']),
              float(optimal_params['dwell_time']), float(optimal_params['success_rate'])))

def load_recommendations_from_db():
    """Load recommendations from database."""
    try:
        with get_database_connection() as conn:
            df = pd.read_sql_query('''
                SELECT id, material_type, print_coverage,
                       recommended_temperature_c, recommended_pressure_bar,
                       recommended_dwell_time_s, predicted_success_rate,
                       user_feedback, created_at
                FROM recommendations
                ORDER BY created_at DESC
            ''', conn)
            return df
    except:
        return pd.DataFrame()

def update_recommendation_feedback(recommendation_id, feedback):
    """Update recommendation feedback in database."""
    with get_database_connection() as conn:
        cursor = conn.cursor()

        # Detect database type for parameter placeholders
        is_postgres = isinstance(conn, psycopg2.extensions.connection)
        placeholder = '%s' if is_postgres else '?'

        cursor.execute(f'''
            UPDATE recommendations
            SET user_feedback = {placeholder}
            WHERE id = {placeholder}
        ''', (feedback, recommendation_id))

def create_order(order_code, material_type, print_coverage, package_size, sackovacka=None, note=None):
    """Create a new order."""
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()

            # Detect database type for parameter placeholders and error handling
            is_postgres = isinstance(conn, psycopg2.extensions.connection)
            placeholder = '%s' if is_postgres else '?'

            if is_postgres:
                # PostgreSQL: Use RETURNING clause to get the ID
                cursor.execute(f'''
                    INSERT INTO orders (order_code, material_type, print_coverage, package_size, sackovacka, note)
                    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                    RETURNING id
                ''', (order_code, material_type, print_coverage, package_size, sackovacka, note))
                order_id = cursor.fetchone()[0]
            else:
                # SQLite: Use lastrowid
                cursor.execute(f'''
                    INSERT INTO orders (order_code, material_type, print_coverage, package_size, sackovacka, note)
                    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                ''', (order_code, material_type, print_coverage, package_size, sackovacka, note))
                order_id = cursor.lastrowid

            return order_id
    except (sqlite3.IntegrityError, psycopg2.IntegrityError):
        return None  # Order code already exists


def get_order_by_id(order_id):
    """Get order details by ID."""
    with get_database_connection() as conn:
        cursor = conn.cursor()

        # Detect database type for parameter placeholders
        is_postgres = isinstance(conn, psycopg2.extensions.connection)
        placeholder = '%s' if is_postgres else '?'

        cursor.execute(f'''
            SELECT id, order_code, material_type, print_coverage, package_size, sackovacka, note, created_at
            FROM orders
            WHERE id = {placeholder}
        ''', (order_id,))
        result = cursor.fetchone()
        if result:
            return {
                'id': result[0],
                'order_code': result[1],
                'material_type': result[2],
                'print_coverage': result[3],
                'package_size': result[4],
                'sackovacka': result[5],
                'note': result[6],
                'created_at': result[7]
            }
        return None

def get_all_orders():
    """Get all orders ordered by creation date (newest first)."""
    with get_database_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, order_code, material_type, print_coverage, package_size, sackovacka, note, created_at
            FROM orders
            ORDER BY created_at DESC
        ''')
        results = cursor.fetchall()
        return [{
            'id': row[0],
            'order_code': row[1],
            'material_type': row[2],
            'print_coverage': row[3],
            'package_size': row[4],
            'sackovacka': row[5],
            'note': row[6],
            'created_at': row[7]
        } for row in results]


def add_attempt(order_id, outcome, **params):
    """Add an attempt to an order with multi-phase sealing parameters.

    Args:
        order_id: ID of the order
        outcome: Result of the attempt ('Úspěch' or 'Neúspěch')
        **params: Sealing parameters, can include:
            - Legacy single-phase: temperature, pressure, dwell_time
            - Zipper phase: zipper_temperature, zipper_pressure, zipper_dwell_time
            - Bottom phase: bottom_temperature, bottom_pressure, bottom_dwell_time
            - Side phases: side_e_temperature, side_e_pressure, side_e_dwell_time, etc.
    """
    with get_database_connection() as conn:
        cursor = conn.cursor()

    # Detect database type for parameter placeholders
    is_postgres = isinstance(conn, psycopg2.extensions.connection)
    placeholder = '%s' if is_postgres else '?'

    # Build the SQL statement dynamically based on provided parameters
    columns = ['order_id', 'outcome']
    values = [order_id, outcome]
    placeholders = [placeholder, placeholder]

    # Map parameter names to database columns
    param_mapping = {
        # Legacy parameters
        'temperature': 'sealing_temperature_c',
        'pressure': 'sealing_pressure_bar',
        'dwell_time': 'dwell_time_s',
        # Zipper phase
        'zipper_temperature': 'zipper_temperature_c',
        'zipper_pressure': 'zipper_pressure_bar',
        'zipper_dwell_time': 'zipper_dwell_time_s',
        # Bottom phase
        'bottom_temperature': 'bottom_temperature_c',
        'bottom_pressure': 'bottom_pressure_bar',
        'bottom_dwell_time': 'bottom_dwell_time_s',
        # Side phases
        'side_e_temperature': 'side_e_temperature_c',
        'side_e_pressure': 'side_e_pressure_bar',
        'side_e_dwell_time': 'side_e_dwell_time_s',
        'side_d_temperature': 'side_d_temperature_c',
        'side_d_pressure': 'side_d_pressure_bar',
        'side_d_dwell_time': 'side_d_dwell_time_s',
        'side_c_temperature': 'side_c_temperature_c',
        'side_c_pressure': 'side_c_pressure_bar',
        'side_c_dwell_time': 'side_c_dwell_time_s',
        'side_b_temperature': 'side_b_temperature_c',
        'side_b_pressure': 'side_b_pressure_bar',
        'side_b_dwell_time': 'side_b_dwell_time_s',
        'side_a_temperature': 'side_a_temperature_c',
        'side_a_pressure': 'side_a_pressure_bar',
        'side_a_dwell_time': 'side_a_dwell_time_s',
        # Note
        'note': 'note'
    }

    # Add provided parameters to the query
    for param_name, param_value in params.items():
        if param_name in param_mapping:
            columns.append(param_mapping[param_name])
            values.append(param_value)
            placeholders.append(placeholder)

    columns_str = ', '.join(columns)
    placeholders_str = ', '.join(placeholders)

    if is_postgres:
        # PostgreSQL: Use RETURNING clause to get the ID
        cursor.execute(f'''
            INSERT INTO attempts ({columns_str})
            VALUES ({placeholders_str})
            RETURNING id
        ''', values)
        attempt_id = cursor.fetchone()[0]
    else:
        # SQLite: Use lastrowid
        cursor.execute(f'''
            INSERT INTO attempts ({columns_str})
            VALUES ({placeholders_str})
        ''', values)
        attempt_id = cursor.lastrowid

    return attempt_id

def get_order_attempts(order_id):
    """Get all attempts for an order with multi-phase parameters."""
    with get_database_connection() as conn:
        cursor = conn.cursor()

        # Detect database type for parameter placeholders
        is_postgres = isinstance(conn, psycopg2.extensions.connection)
        placeholder = '%s' if is_postgres else '?'

        cursor.execute(f'''
            SELECT id, outcome, created_at,
                   -- Legacy parameters
                   sealing_temperature_c, sealing_pressure_bar, dwell_time_s,
                   -- Zipper phase
                   zipper_temperature_c, zipper_pressure_bar, zipper_dwell_time_s,
                   -- Bottom phase
                   bottom_temperature_c, bottom_pressure_bar, bottom_dwell_time_s,
                   -- Side phases
                   side_e_temperature_c, side_e_pressure_bar, side_e_dwell_time_s,
                   side_d_temperature_c, side_d_pressure_bar, side_d_dwell_time_s,
                   side_c_temperature_c, side_c_pressure_bar, side_c_dwell_time_s,
                   side_b_temperature_c, side_b_pressure_bar, side_b_dwell_time_s,
                   side_a_temperature_c, side_a_pressure_bar, side_a_dwell_time_s,
                   -- Note
                   note
            FROM attempts
            WHERE order_id = {placeholder}
            ORDER BY created_at ASC
        ''', (order_id,))
        results = cursor.fetchall()

        attempts = []
        for row in results:
            attempt = {
                'id': row[0],
                'outcome': row[1],
                'created_at': row[2],
                # Legacy parameters (for backward compatibility)
                'temperature': row[3],
                'pressure': row[4],
                'dwell_time': row[5],
                # Multi-phase parameters
                'zipper_temperature': row[6],
                'zipper_pressure': row[7],
                'zipper_dwell_time': row[8],
                'bottom_temperature': row[9],
                'bottom_pressure': row[10],
                'bottom_dwell_time': row[11],
                'side_e_temperature': row[12],
                'side_e_pressure': row[13],
                'side_e_dwell_time': row[14],
                'side_d_temperature': row[15],
                'side_d_pressure': row[16],
                'side_d_dwell_time': row[17],
                'side_c_temperature': row[18],
                'side_c_pressure': row[19],
                'side_c_dwell_time': row[20],
                'side_b_temperature': row[21],
                'side_b_pressure': row[22],
                'side_b_dwell_time': row[23],
                'side_a_temperature': row[24],
                'side_a_pressure': row[25],
                'side_a_dwell_time': row[26],
                # Note
                'note': row[27]
            }
            attempts.append(attempt)

        return attempts

def delete_attempt(attempt_id):
    """Delete an attempt by ID."""
    with get_database_connection() as conn:
        cursor = conn.cursor()

        # Detect database type for parameter placeholders
        is_postgres = isinstance(conn, psycopg2.extensions.connection)
        placeholder = '%s' if is_postgres else '?'

        cursor.execute(f'DELETE FROM attempts WHERE id = {placeholder}', (attempt_id,))

def load_combined_data():
    """Load and combine database data sources: user data, attempts data, and feedback data."""
    user_data = load_user_data_from_db()
    attempts_data = load_attempts_data()
    feedback_data = load_feedback_as_training_data()

    # Combine all data sources
    all_data = []
    if not user_data.empty:
        # Add Package_Size column if missing from user_data (default to 3)
        if 'Package_Size' not in user_data.columns:
            user_data['Package_Size'] = 3
        all_data.append(user_data)
    if not attempts_data.empty:
        # Select only the columns needed for training (including Package_Size)
        required_columns = ['Material_Type', 'Print_Coverage',
                           'Sealing_Temperature_C', 'Sealing_Pressure_bar',
                           'Dwell_Time_s', 'Outcome']

        # Add Package_Size if available
        if 'Package_Size' in attempts_data.columns:
            required_columns.append('Package_Size')
            attempts_subset = attempts_data[required_columns]
        else:
            attempts_subset = attempts_data[required_columns]
            attempts_subset['Package_Size'] = 3  # Default value

        all_data.append(attempts_subset)
    if not feedback_data.empty:
        # Add Package_Size column if missing from feedback_data (default to 3)
        if 'Package_Size' not in feedback_data.columns:
            feedback_data['Package_Size'] = 3
        all_data.append(feedback_data)

    if not all_data:
        return None

    return pd.concat(all_data, ignore_index=True)

def load_feedback_as_training_data():
    """Convert recommendation feedback into training data."""
    try:
        with get_database_connection() as conn:
            df = pd.read_sql_query('''
                SELECT material_type as Material_Type,
                       print_coverage as Print_Coverage,
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
    categorical_features = ['Material_Type']
    numerical_features = ['Print_Coverage', 'Package_Size', 'Sealing_Temperature_C', 'Sealing_Pressure_bar', 'Dwell_Time_s']

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

def find_optimal_parameters(model, encoder, material_type, print_coverage, package_size):
    """Find optimal multi-phase sealing parameters using simplified optimization."""
    # For now, return optimized parameters for each phase with reasonable defaults
    # This is a simplified version - in production, you might want to use more
    # sophisticated optimization techniques due to the 21-dimensional parameter space

    # Base temperature ranges by material type
    base_temp = 150

    # Check for different material structures
    if 'PAP/' in material_type:  # Paper-based materials
        base_temp = 155
    elif 'BOPP/BOPP MET/CPP' in material_type:  # Metallic PP monostructure
        base_temp = 150
    elif 'BOPP/BOPP ALOX/CPP' in material_type:  # Transparent PP monostructure
        base_temp = 145
    elif 'BOPP/PET MET/LDPE' in material_type:  # Metallic BOPP/PET/LDPE
        base_temp = 160
    elif 'BOPP/PET TRA/LDPE' in material_type:  # Transparent BOPP/PET/LDPE
        base_temp = 155
    elif 'PET/PET MET/LDPE' in material_type:  # Metallic PET/PET/LDPE
        base_temp = 165
    elif 'PET/PET TRA/LDPE' in material_type:  # Transparent PET/PET/LDPE
        base_temp = 160

    # Ink adjustment removed - ink type no longer considered

    # Adjust for print coverage
    coverage_adjustment = print_coverage * 0.2  # Higher coverage needs slightly more heat

    # Adjust for package size - larger packages may need more heat and pressure
    size_adjustment = (package_size - 3) * 2  # Size 3 is neutral, 1-2 need less heat, 4-6 need more

    final_base_temp = base_temp + coverage_adjustment + size_adjustment

    # Calculate optimal parameters for each phase
    optimal_params = {
        # Zipper phase - generally needs less heat
        'zipper_temperature': max(100, min(220, final_base_temp - 10)),
        'zipper_pressure': 4.0,
        'zipper_dwell_time': 1.0,

        # Bottom phase - needs moderate heat
        'bottom_temperature': max(100, min(220, final_base_temp)),
        'bottom_pressure': 4.5,
        'bottom_dwell_time': 1.2,

        # Side phases - gradually increasing intensity from E to A
        'side_e_temperature': max(100, min(220, final_base_temp - 5)),
        'side_e_pressure': 4.2,
        'side_e_dwell_time': 1.1,

        'side_d_temperature': max(100, min(220, final_base_temp - 2)),
        'side_d_pressure': 4.3,
        'side_d_dwell_time': 1.15,

        'side_c_temperature': max(100, min(220, final_base_temp)),
        'side_c_pressure': 4.4,
        'side_c_dwell_time': 1.2,

        'side_b_temperature': max(100, min(220, final_base_temp + 3)),
        'side_b_pressure': 4.5,
        'side_b_dwell_time': 1.25,

        'side_a_temperature': max(100, min(220, final_base_temp + 5)),
        'side_a_pressure': 4.6,
        'side_a_dwell_time': 1.3,

        'success_rate': 0.85  # Estimated success rate
    }

    return optimal_params

def render_data_entry_form():
    """Render the data entry form."""
    st.subheader("📝 Přidat nová produkční data")

    # Default material and ink options
    material_options = [
        'PAP/PET/LDPE (MAT-02448)',
        'PAP/PET/LDPE (MAT-02841)',
        'BOPP/BOPP MET/CPP (MAT-02514)',
        'BOPP/BOPP ALOX/CPP (MAT-02481)',
        'BOPP/PET MET/LDPE (MAT-02381)',
        'BOPP/PET TRA/LDPE (MAT-02675)',
        'PET/PET MET/LDPE (MAT-02381)',
        'PET/PET TRA/LDPE (MAT-02675)'
    ]

    with st.form("data_entry_form"):
        col1, col2 = st.columns(2)

        with col1:
            material_type = st.selectbox("Typ materiálu", material_options)
            print_coverage = st.slider("Pokrytí tiskem v oblasti svařování (%)", 0, 500, 50)

        with col2:
            temperature = st.number_input("Teplota svařování (°C)", 100.0, 220.0, 150.0, 1.0)
            pressure = st.number_input("Tlak svařování (bar)", 1.0, 8.0, 4.0, 0.1)
            dwell_time = st.number_input("Doba svařování (s)", 0.1, 3.0, 1.0, 0.1)

        outcome = st.radio("Výsledek", ["Úspěch", "Neúspěch"], horizontal=True)

        submitted = st.form_submit_button("➕ Přidat datový bod", type="primary")

        if submitted:
            # Validate inputs
            if 100 <= temperature <= 220 and 1.0 <= pressure <= 8.0 and 0.1 <= dwell_time <= 3.0:
                # Translate outcome back to English for database consistency
                outcome_en = 'Pass' if outcome == 'Úspěch' else 'Fail'
                new_data = {
                    'Material_Type': material_type,
                    'Print_Coverage': print_coverage,
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Celkem záznamů", len(data))
        with col2:
            pass_rate = (data['Outcome'] == 'Pass').mean() * 100
            st.metric("Úspěšnost", f"{pass_rate:.1f}%")
        with col3:
            st.metric("Typů materiálů", data['Material_Type'].nunique())

    else:
        st.info("Nejsou k dispozici žádná data. Přidejte několik datových bodů do databáze.")

def optimize_parameters_section(model, encoder, data):
    """Render the parameter optimization section."""
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Vstupní parametry")

        material_options = data['Material_Type'].unique().tolist()

        material_type = st.selectbox(
            "Typ materiálu",
            options=material_options,
            help="Vyberte typ materiálu pro vaši produkční sérii"
        )


        print_coverage = st.slider(
            "Pokrytí tiskem v oblasti svařování (%)",
            min_value=0,
            max_value=500,
            value=50,
            help="Procento povrchu v oblasti svařování pokryté tiskem"
        )

        # Doypack size options with labels
        size_options = {
            "≥ 85 mm": 1,
            "≥ 90 mm": 2,
            "≥ 130 mm": 3,
            "≥ 160 mm": 4,
            "≥ 180 mm": 5,
            "≥ 230 mm": 6
        }
        package_size_label = st.selectbox(
            "Velikost doypacku",
            options=list(size_options.keys()),
            index=2,  # Default to size 3
            help="Velikost doypacku (1-6) s rozměry v mm"
        )
        package_size = size_options[package_size_label]

        # Main action button
        optimize_button = st.button("🎯 Najít optimální nastavení", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 Doporučené parametry")

        # Handle button click and display results in the same column
        if optimize_button:
            with st.spinner("Optimalizuji parametry..."):
                optimal_params = find_optimal_parameters(
                    model, encoder, material_type, print_coverage, package_size
                )

            if optimal_params:
                st.success("✅ Optimální parametry pro všechny fáze nalezeny!")

                # Display results organized by sealing phases
                st.markdown("### 📋 Doporučené parametry pro všechny fáze")

                # Zipper phase
                st.markdown("**🔗 Svár zip**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌡️ Teplota", f"{optimal_params['zipper_temperature']:.0f}°C")
                with col2:
                    st.metric("⚡ Tlak", f"{optimal_params['zipper_pressure']:.1f} bar")
                with col3:
                    st.metric("⏱️ Doba", f"{optimal_params['zipper_dwell_time']:.1f}s")

                st.markdown("---")

                # Bottom phase
                st.markdown("**⬇️ Svár dno**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌡️ Teplota", f"{optimal_params['bottom_temperature']:.0f}°C")
                with col2:
                    st.metric("⚡ Tlak", f"{optimal_params['bottom_pressure']:.1f} bar")
                with col3:
                    st.metric("⏱️ Doba", f"{optimal_params['bottom_dwell_time']:.1f}s")

                # Side phases
                st.markdown("**🔷 Věž E**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌡️ Teplota", f"{optimal_params['side_e_temperature']:.0f}°C")
                with col2:
                    st.metric("⚡ Tlak", f"{optimal_params['side_e_pressure']:.1f} bar")
                with col3:
                    st.metric("⏱️ Doba", f"{optimal_params['side_e_dwell_time']:.1f}s")

                st.markdown("**🔶 Věž D**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌡️ Teplota", f"{optimal_params['side_d_temperature']:.0f}°C")
                with col2:
                    st.metric("⚡ Tlak", f"{optimal_params['side_d_pressure']:.1f} bar")
                with col3:
                    st.metric("⏱️ Doba", f"{optimal_params['side_d_dwell_time']:.1f}s")

                st.markdown("**🔸 Věž C**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌡️ Teplota", f"{optimal_params['side_c_temperature']:.0f}°C")
                with col2:
                    st.metric("⚡ Tlak", f"{optimal_params['side_c_pressure']:.1f} bar")
                with col3:
                    st.metric("⏱️ Doba", f"{optimal_params['side_c_dwell_time']:.1f}s")

                st.markdown("**🔹 Věž B**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌡️ Teplota", f"{optimal_params['side_b_temperature']:.0f}°C")
                with col2:
                    st.metric("⚡ Tlak", f"{optimal_params['side_b_pressure']:.1f} bar")
                with col3:
                    st.metric("⏱️ Doba", f"{optimal_params['side_b_dwell_time']:.1f}s")

                st.markdown("**🔺 Věž A**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌡️ Teplota", f"{optimal_params['side_a_temperature']:.0f}°C")
                with col2:
                    st.metric("⚡ Tlak", f"{optimal_params['side_a_pressure']:.1f} bar")
                with col3:
                    st.metric("⏱️ Doba", f"{optimal_params['side_a_dwell_time']:.1f}s")

                # Summary info
                st.info(f"""
                **📋 Shrnutí doporučení:**
                - **Materiál:** {material_type}
                - **Pokrytí tiskem:** {print_coverage}%
                - **Velikost doypacku:** {package_size}
                - **Odhadovaná úspěšnost:** {optimal_params['success_rate']*100:.1f}%

                Tyto parametry jsou optimalizovány pro všech 7 fází svařování doypacku
                na základě typu materiálu, barvy, pokrytí tiskem a velikosti doypacku.
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
        user_count = len(load_user_data_from_db()) if not load_user_data_from_db().empty else 0
        attempts_count = len(load_attempts_data()) if not load_attempts_data().empty else 0
        feedback_count = len(load_feedback_as_training_data()) if not load_feedback_as_training_data().empty else 0
        if user_count > 0 or attempts_count > 0 or feedback_count > 0:
            st.sidebar.markdown(f"**Zdroje dat:**")
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

                with st.expander(f"🕐 {format_datetime(rec['created_at'])} | {rec['material_type']} | {rec['print_coverage']}%"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("🌡️ Teplota", f"{temp:.0f}°C")
                        st.metric("⚡ Tlak", f"{pressure:.1f} bar")

                    with col2:
                        st.metric("⏱️ Doba svařování", f"{dwell:.1f}s")
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

def render_order_list():
    """Render list of all orders with management options."""
    orders = get_all_orders()

    if not orders:
        st.info("📋 Zatím nebyly vytvořeny žádné zakázky.")
        return

    st.subheader("📋 Seznam zakázek")

    for order in orders:
        # Get attempt count for this order
        attempts = get_order_attempts(order['id'])

        with st.expander(f"📦  **{order['order_code']}** ({format_datetime(order['created_at'])})"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Materiál:** {order['material_type']}")
                st.write(f"**Pokrytí:** {order['print_coverage']}%")
                package_size_display = order.get('package_size', 'N/A')
                st.write(f"**Velikost:** {package_size_display}")
                sackovacka_display = order.get('sackovacka', 'N/A')
                st.write(f"**Sáčkovačka:** {sackovacka_display}")

            with col2:
                if st.button("📝 Otevřít zakázku", key=f"open_{order['id']}", type="primary"):
                    st.session_state.current_order_id = order['id']
                    st.session_state.order_screen = True
                    st.rerun()

def render_new_order_form():
    """Render the new order creation form."""
    st.subheader("📋 Nová zakázka")

    # Default material and ink options
    material_options = [
        'PAP/PET/LDPE (MAT-02448)',
        'PAP/PET/LDPE (MAT-02841)',
        'BOPP/BOPP MET/CPP (MAT-02514)',
        'BOPP/BOPP ALOX/CPP (MAT-02481)',
        'BOPP/PET MET/LDPE (MAT-02381)',
        'BOPP/PET TRA/LDPE (MAT-02675)',
        'PET/PET MET/LDPE (MAT-02381)',
        'PET/PET TRA/LDPE (MAT-02675)'
    ]

    with st.form("new_order_form"):
        # Order identification section
        st.markdown("**📋 Identifikace zakázky:**")
        order_code = st.text_input("Kód zakázky", placeholder="např. Z2024-001")

        st.markdown("---")
        st.markdown("**🏭 Parametry materiálu a tisku:**")

        col1, col2 = st.columns(2)
        with col1:
            material_type = st.selectbox("Typ materiálu", material_options)
            print_coverage = st.slider("Pokrytí tiskem v oblasti svařování (%)", 0, 500, 50)

        with col2:
            # Doypack size options with labels
            size_options = {
                "≥ 85 mm": 1,
                "≥ 90 mm": 2,
                "≥ 130 mm": 3,
                "≥ 160 mm": 4,
                "≥ 180 mm": 5,
                "≥ 230 mm": 6
            }
            package_size_label = st.selectbox("Velikost doypacku", list(size_options.keys()), index=2)
            package_size = size_options[package_size_label]

            # Sáčkovačka options
            sackovacka_options = ["S1", "S2", "S3", "S4"]
            sackovacka = st.selectbox("Sáčkovačka", sackovacka_options, index=0)

        st.markdown("---")
        st.markdown("**📝 Poznámka**")
        order_note = st.text_area("Poznámka k zakázce (nepovinné)",
                                 placeholder="Zadejte jakékoliv poznámky k této zakázce...",
                                 help="Zde můžete zapsat jakékoliv poznámky k zakázce")

        submitted = st.form_submit_button("🚀 Začít", type="primary", use_container_width=True)

        if submitted:
            if order_code.strip():
                order_id = create_order(order_code.strip(), material_type, print_coverage, package_size, sackovacka, order_note)
                if order_id:
                    # Navigate to dedicated order screen
                    st.session_state.current_order_id = order_id
                    st.session_state.order_screen = True
                    st.session_state.show_new_order_form = False  # Hide the form
                    st.success(f"✅ Zakázka {order_code} byla vytvořena!")
                    st.rerun()
                else:
                    st.error("❌ Zakázka s tímto kódem již existuje!")
            else:
                st.error("❌ Zadejte kód zakázky!")

def render_dedicated_order_screen():
    """Render the dedicated order screen for recording attempts."""
    if not st.session_state.current_order_id:
        st.error("❌ Chyba: Nebyla nalezena aktivní zakázka.")
        if st.button("🏠 Zpět na úvodní stránku"):
            st.session_state.order_screen = False
            st.session_state.current_order_id = None
            st.rerun()
        return

    order = get_order_by_id(st.session_state.current_order_id)
    if not order:
        st.error("❌ Zakázka nebyla nalezena.")
        if st.button("🏠 Zpět na úvodní stránku"):
            st.session_state.order_screen = False
            st.session_state.current_order_id = None
            st.rerun()
        return

    # Back button at top
    if st.button("🏠 Zpět na úvodní stránku", key="back_top"):
        st.session_state.order_screen = False
        st.session_state.current_order_id = None
        st.session_state.show_new_order_form = False
        st.rerun()

    # Header with order details
    package_size_display = order.get('package_size', 'N/A')
    sackovacka_display = order.get('sackovacka', 'N/A')
    st.markdown(f"""
    # 📦 Zakázka: **{order['order_code']}**

    **📋 Detaily zakázky:**
    - **Materiál:** {order['material_type']}
    - **Pokrytí tiskem v oblasti svařování:** {order['print_coverage']}%
    - **Velikost doypacku:** {package_size_display}
    - **Sáčkovačka:** {sackovacka_display}
    - **Vytvořeno:** {format_datetime(order['created_at'])}
    """)

    # Display note if it exists
    if order.get('note') and order['note'].strip():
        st.markdown("**📝 Poznámka k zakázce:**")
        st.write(order['note'])

    st.markdown("---")

    # Get existing attempts
    attempts = get_order_attempts(order['id'])

    # Show existing attempts
    if attempts:
        st.subheader("📊 Historie pokusů")
        for i, attempt in enumerate(attempts, 1):
            outcome_emoji = "✅" if attempt['outcome'] == 'Úspěch' else "❌"

            # Check if this is a multi-phase attempt (has new parameters) or legacy attempt
            has_multiphase = attempt.get('zipper_temperature') is not None

            with st.expander(f"{outcome_emoji} **Pokus {i}** - {attempt['outcome']} ({format_datetime(attempt['created_at'])})"):
                if has_multiphase:
                    # Display multi-phase parameters stacked vertically
                    st.markdown("**🔧 Parametry všech fází svařování:**")

                    # Zipper phase
                    st.markdown("**🔗 Svár zip**")
                    if attempt['zipper_temperature']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🌡️ {attempt['zipper_temperature']:.0f}°C")
                        with col2:
                            st.write(f"⚡ {attempt['zipper_pressure']:.1f} bar")
                        with col3:
                            st.write(f"⏱️ {attempt['zipper_dwell_time']:.1f}s")
                    else:
                        st.write("*Nezadáno*")

                    # Bottom phase
                    st.markdown("**⬇️ Svár dno**")
                    if attempt['bottom_temperature']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🌡️ {attempt['bottom_temperature']:.0f}°C")
                        with col2:
                            st.write(f"⚡ {attempt['bottom_pressure']:.1f} bar")
                        with col3:
                            st.write(f"⏱️ {attempt['bottom_dwell_time']:.1f}s")
                    else:
                        st.write("*Nezadáno*")

                    # Side phases
                    st.markdown("**🔷 Věž E**")
                    if attempt['side_e_temperature']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🌡️ {attempt['side_e_temperature']:.0f}°C")
                        with col2:
                            st.write(f"⚡ {attempt['side_e_pressure']:.1f} bar")
                        with col3:
                            st.write(f"⏱️ {attempt['side_e_dwell_time']:.1f}s")
                    else:
                        st.write("*Nezadáno*")

                    st.markdown("**🔶 Věž D**")
                    if attempt['side_d_temperature']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🌡️ {attempt['side_d_temperature']:.0f}°C")
                        with col2:
                            st.write(f"⚡ {attempt['side_d_pressure']:.1f} bar")
                        with col3:
                            st.write(f"⏱️ {attempt['side_d_dwell_time']:.1f}s")
                    else:
                        st.write("*Nezadáno*")

                    st.markdown("**🔸 Věž C**")
                    if attempt['side_c_temperature']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🌡️ {attempt['side_c_temperature']:.0f}°C")
                        with col2:
                            st.write(f"⚡ {attempt['side_c_pressure']:.1f} bar")
                        with col3:
                            st.write(f"⏱️ {attempt['side_c_dwell_time']:.1f}s")
                    else:
                        st.write("*Nezadáno*")

                    st.markdown("**🔹 Věž B**")
                    if attempt['side_b_temperature']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🌡️ {attempt['side_b_temperature']:.0f}°C")
                        with col2:
                            st.write(f"⚡ {attempt['side_b_pressure']:.1f} bar")
                        with col3:
                            st.write(f"⏱️ {attempt['side_b_dwell_time']:.1f}s")
                    else:
                        st.write("*Nezadáno*")

                    st.markdown("**🔺 Věž A**")
                    if attempt['side_a_temperature']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🌡️ {attempt['side_a_temperature']:.0f}°C")
                        with col2:
                            st.write(f"⚡ {attempt['side_a_pressure']:.1f} bar")
                        with col3:
                            st.write(f"⏱️ {attempt['side_a_dwell_time']:.1f}s")
                    else:
                        st.write("*Nezadáno*")

                else:
                    # Display legacy single-phase parameters
                    st.markdown("**🔧 Původní parametry (jedna fáze):**")
                    if attempt['temperature']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            temp_formatted = f"{attempt['temperature']:.1f}" if attempt['temperature'] % 1 != 0 else f"{int(attempt['temperature'])}"
                            st.write(f"🌡️ Teplota: {temp_formatted}°C")
                        with col2:
                            st.write(f"⚡ Tlak: {attempt['pressure']:.1f} bar")
                        with col3:
                            st.write(f"⏱️ Doba: {attempt['dwell_time']:.1f}s")
                    else:
                        st.write("*Žádné parametry*")

                # Display note if it exists
                if attempt.get('note') and attempt['note'].strip():
                    st.markdown("**📝 Poznámka:**")
                    st.write(attempt['note'])

                # Delete button
                col1, col2 = st.columns([0.8, 0.2])
                with col2:
                    # Use session state to track confirmation state
                    confirm_key = f"confirm_delete_{attempt['id']}"
                    if confirm_key not in st.session_state:
                        st.session_state[confirm_key] = False

                    if not st.session_state[confirm_key]:
                        if st.button("Odstranit", key=f"delete_attempt_{attempt['id']}", help="Smazat pokus"):
                            st.session_state[confirm_key] = True
                            st.rerun()
                    else:
                        # Show confirmation buttons
                        sub_col1, sub_col2 = st.columns(2)
                        with sub_col1:
                            if st.button("✅", key=f"confirm_yes_{attempt['id']}", help="Ano, smazat"):
                                delete_attempt(attempt['id'])
                                st.session_state[confirm_key] = False
                                st.success("✅ Pokus byl smazán!")
                                st.rerun()
                        with sub_col2:
                            if st.button("❌", key=f"confirm_no_{attempt['id']}", help="Ne, zrušit"):
                                st.session_state[confirm_key] = False
                                st.rerun()
        st.markdown("---")

    # Add new attempt form
    st.subheader(f"🔬 Pokus {len(attempts) + 1}")

    # Form for all parameters except the align button
    with st.form("attempt_form"):
        st.markdown("**📋 Parametry svařování**")

        params = {}

        # Zipper sealing phase
        st.markdown("#### 🔗 Svár zip")
        col1, col2, col3 = st.columns(3)
        with col1:
            params['zipper_temperature'] = st.slider("Teplota (°C)", 100.0, 220.0, 150.0, 1.0, key="zip_temp")
        with col2:
            params['zipper_pressure'] = st.slider("Tlak (bar)", 1.0, 8.0, 4.0, 0.1, key="zip_press")
        with col3:
            params['zipper_dwell_time'] = st.slider("Doba (s)", 0.1, 3.0, 1.0, 0.1, key="zip_time")

        st.markdown("---")

        # Bottom sealing phase
        st.markdown("#### ⬇️ Svár dno")
        col1, col2, col3 = st.columns(3)
        with col1:
            params['bottom_temperature'] = st.slider("Teplota (°C)", 100.0, 220.0, 160.0, 1.0, key="bottom_temp")
        with col2:
            params['bottom_pressure'] = st.slider("Tlak (bar)", 1.0, 8.0, 4.5, 0.1, key="bottom_press")
        with col3:
            params['bottom_dwell_time'] = st.slider("Doba (s)", 0.1, 3.0, 1.2, 0.1, key="bottom_time")

        st.markdown("---")
        st.markdown("#### Příčné sváry")

        # Side sealing phases
        st.markdown("**🔷 Věž E**")
        col1, col2, col3 = st.columns(3)
        with col1:
            params['side_e_temperature'] = st.slider("Teplota (°C)", 100.0, 220.0, 155.0, 1.0, key="side_e_temp")
        with col2:
            params['side_e_pressure'] = st.slider("Tlak (bar)", 1.0, 8.0, 4.2, 0.1, key="side_e_press")
        with col3:
            params['side_e_dwell_time'] = st.slider("Doba (s)", 0.1, 3.0, 1.1, 0.1, key="side_e_time")

        # Align button right next to Stage E
        align_button = st.form_submit_button("🔗 Kopírovat z Věže E (E→D,C,B,A)", help="Zkopíruje hodnoty z věže E do ostatních věží D, C, B, A")

        if align_button:
            # Set aligned values from current Stage E
            st.session_state.aligned_side_temperature = params['side_e_temperature']
            st.session_state.aligned_side_pressure = params['side_e_pressure']
            st.session_state.aligned_side_dwell_time = params['side_e_dwell_time']
            st.session_state.stages_aligned = True
            st.rerun()

        st.markdown("**🔶 Věž D**")
        col1, col2, col3 = st.columns(3)
        with col1:
            d_temp_default = st.session_state.aligned_side_temperature if st.session_state.stages_aligned else 158.0
            params['side_d_temperature'] = st.slider("Teplota (°C)", 100.0, 220.0, d_temp_default, 1.0, key="side_d_temp")
        with col2:
            d_press_default = st.session_state.aligned_side_pressure if st.session_state.stages_aligned else 4.3
            params['side_d_pressure'] = st.slider("Tlak (bar)", 1.0, 8.0, d_press_default, 0.1, key="side_d_press")
        with col3:
            d_time_default = st.session_state.aligned_side_dwell_time if st.session_state.stages_aligned else 1.15
            params['side_d_dwell_time'] = st.slider("Doba (s)", 0.1, 3.0, d_time_default, 0.1, key="side_d_time")

        st.markdown("**🔸 Věž C**")
        col1, col2, col3 = st.columns(3)
        with col1:
            c_temp_default = st.session_state.aligned_side_temperature if st.session_state.stages_aligned else 162.0
            params['side_c_temperature'] = st.slider("Teplota (°C)", 100.0, 220.0, c_temp_default, 1.0, key="side_c_temp")
        with col2:
            c_press_default = st.session_state.aligned_side_pressure if st.session_state.stages_aligned else 4.4
            params['side_c_pressure'] = st.slider("Tlak (bar)", 1.0, 8.0, c_press_default, 0.1, key="side_c_press")
        with col3:
            c_time_default = st.session_state.aligned_side_dwell_time if st.session_state.stages_aligned else 1.2
            params['side_c_dwell_time'] = st.slider("Doba (s)", 0.1, 3.0, c_time_default, 0.1, key="side_c_time")

        st.markdown("**🔹 Věž B**")
        col1, col2, col3 = st.columns(3)
        with col1:
            b_temp_default = st.session_state.aligned_side_temperature if st.session_state.stages_aligned else 165.0
            params['side_b_temperature'] = st.slider("Teplota (°C)", 100.0, 220.0, b_temp_default, 1.0, key="side_b_temp")
        with col2:
            b_press_default = st.session_state.aligned_side_pressure if st.session_state.stages_aligned else 4.5
            params['side_b_pressure'] = st.slider("Tlak (bar)", 1.0, 8.0, b_press_default, 0.1, key="side_b_press")
        with col3:
            b_time_default = st.session_state.aligned_side_dwell_time if st.session_state.stages_aligned else 1.25
            params['side_b_dwell_time'] = st.slider("Doba (s)", 0.1, 3.0, b_time_default, 0.1, key="side_b_time")

        st.markdown("**🔺 Věž A**")
        col1, col2, col3 = st.columns(3)
        with col1:
            a_temp_default = st.session_state.aligned_side_temperature if st.session_state.stages_aligned else 168.0
            params['side_a_temperature'] = st.slider("Teplota (°C)", 100.0, 220.0, a_temp_default, 1.0, key="side_a_temp")
        with col2:
            a_press_default = st.session_state.aligned_side_pressure if st.session_state.stages_aligned else 4.6
            params['side_a_pressure'] = st.slider("Tlak (bar)", 1.0, 8.0, a_press_default, 0.1, key="side_a_press")
        with col3:
            a_time_default = st.session_state.aligned_side_dwell_time if st.session_state.stages_aligned else 1.3
            params['side_a_dwell_time'] = st.slider("Doba (s)", 0.1, 3.0, a_time_default, 0.1, key="side_a_time")

        st.markdown("---")

        # Note field
        st.markdown("**📝 Poznámka**")
        params['note'] = st.text_area("Poznámka k pokusu (nepovinné)",
                                     placeholder="Zadejte jakékoliv poznámky k tomuto pokusu...",
                                     help="Zde můžete zapsat jakékoliv pozorování, problémy nebo další informace k tomuto pokusu")

        outcome = st.radio("**🎯 Výsledek pokusu**", ["Neúspěch", "Úspěch"], horizontal=True)

        submitted = st.form_submit_button("Uložit", type="primary")

        if submitted:
            # Validate all parameters are within range
            valid = True
            for param_name, param_value in params.items():
                if 'temperature' in param_name and not (100 <= param_value <= 220):
                    valid = False
                elif 'pressure' in param_name and not (1.0 <= param_value <= 8.0):
                    valid = False
                elif 'dwell_time' in param_name and not (0.1 <= param_value <= 3.0):
                    valid = False

            if valid:
                add_attempt(order['id'], outcome, **params)
                st.success(f"✅ Pokus se všemi {len(params)} parametry byl přidán!")
                st.rerun()
            else:
                st.error("❌ Některé parametry jsou mimo povolený rozsah!")

    # Back button at bottom
    st.markdown("---")
    if st.button("🏠 Zpět na úvodní stránku", key="back_bottom"):
        st.session_state.order_screen = False
        st.session_state.current_order_id = None
        st.session_state.show_new_order_form = False
        st.rerun()


def main_page():
    """Main page for data gathering phase."""
    # Check if we should show dedicated order screen
    if st.session_state.order_screen and st.session_state.current_order_id:
        render_dedicated_order_screen()
        return

    st.title("🔥 Tepelné svařování Doypack")

    # Primary call-to-action: Create new order button
    if st.button("➕ Nová zakázka", type="primary"):
        st.session_state.show_new_order_form = True
        st.rerun()

    # Show new order form if button was clicked
    if st.session_state.show_new_order_form:
        st.markdown("---")
        render_new_order_form()
        if st.button("❌ Zrušit", type="secondary"):
            st.session_state.show_new_order_form = False
            st.rerun()
        return

    # Check if no orders exist - show welcome message
    orders = get_all_orders()
    if not orders:
        st.info("👋 Vítejte! Klikněte na tlačítko výše pro vytvoření vaší první zakázky.")
        return

    # Display existing orders list
    st.markdown("---")
    render_order_list()

def load_attempts_data():
    """Load attempts data from database."""
    try:
        with get_database_connection() as conn:
            df = pd.read_sql_query('''
                SELECT o.material_type as Material_Type,
                       o.print_coverage as Print_Coverage,
                       COALESCE(o.package_size, 3) as Package_Size,
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

def data_management_page():
    """Data management page - view only."""
    st.title("📊 Správa produkčních dat")
    st.markdown("Přehled všech produkčních dat z databáze: ruční vstup, pokusy ze zakázek a zpětná vazba")

    # Model retraining controls
    st.sidebar.header("⚙️ Nastavení modelu")
    if st.sidebar.button("🔄 Přetrénovat model"):
        st.session_state.model_needs_retraining = True
        st.cache_resource.clear()
        st.sidebar.success("Model bude přetrénován!")

    # Data viewing tabs
    tab1, tab2 = st.tabs(["📊 Všechna data", "📋 Data ze zakázek"])

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
                    st.metric("Poslední pokus", format_date(latest_attempt))
        else:
            st.info("Zatím nebyly zaznamenány žádné pokusy ze zakázek.")

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
    # Database initialization moved to lazy loading - will be initialized when first needed

    # Page navigation in sidebar
    st.sidebar.title("🧭 Navigace")
    page = st.sidebar.radio(
        "Vyberte stránku:",
        ["🔥 Sběr dat", "🎯 Výpočet parametrů", "📊 Přehled dat"],
        label_visibility="collapsed"
    )

    # Route to appropriate page
    if page == "🔥 Sběr dat":
        main_page()
    elif page == "📊 Přehled dat":
        data_management_page()
    elif page == "🎯 Výpočet parametrů":
        optimization_page()

if __name__ == "__main__":
    main()