import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lasio
import matplotlib.pyplot as plt
import tempfile
import os

st.title("Formation Permeability Predictor & Multi-Well Viewer")
st.write("Upload one or multiple well log files (`.las`, `.txt`, `.csv`, or `.xlsx`) to predict continuous permeability and visualize tracks.")

# Load saved model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('permeability_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

def standardize_log_columns(df):
    df.columns = [str(col).strip().upper() for col in df.columns]
    mnemonic_aliases = {
        'GR': ['GR', 'GAMMA', 'SGR', 'GRC', 'GRD', 'HSGR'],
        'NPHI': ['NPHI', 'NPH', 'NPOR', 'APTC', 'CNC', 'TNPH'],
        'RHOB': ['RHOB', 'DEN', 'RHOZ', 'ZDEN', 'HDEN', 'DRHO'],
        'RT': ['RT', 'LLD', 'ILD', 'RESD', 'AT90', 'RD'],
        'DEPT': ['DEPT', 'DEPTH', 'MD', 'TDEP']
    }
    rename_mapping = {}
    for standard_name, aliases in mnemonic_aliases.items():
        for col in df.columns:
            if col in aliases:
                rename_mapping[col] = standard_name
                break
    df = df.rename(columns=rename_mapping)
    return df

# File uploader allowing multiple files
uploaded_files = st.file_uploader(
    "Upload Well Data Files", 
    type=["las", "txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    tab_names = [file.name for file in uploaded_files]
    tabs = st.tabs(tab_names)

    for i, uploaded_file in enumerate(uploaded_files):
        with tabs[i]:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                df = None
                if file_extension == 'las':
                    las_obj = lasio.read(tmp_path)
                    df = las_obj.df().reset_index()
                elif file_extension in ['csv', 'txt']:
                    try:
                        df = pd.read_csv(tmp_path)
                    except Exception:
                        df = pd.read_csv(tmp_path, sep='\t')
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(tmp_path)

                if df is not None:
                    df = standardize_log_columns(df)
                    st.success(f"Successfully loaded: {uploaded_file.name}")
                    st.subheader("Raw Data Preview")
                    st.dataframe(df.head())

                    required_cols = ['GR', 'NPHI', 'RHOB', 'RT']
                    missing_cols = [col for col in required_cols if col not in df.columns]

                    if missing_cols:
                        st.warning(f"Could not automatically detect: {missing_cols}. Please map your columns manually:")
                        available_cols = list(df.columns)

                        def get_default_index(target, options):
                            for idx, opt in enumerate(options):
                                if target in opt:
                                    return idx
                            return 0

                        col_gr = st.selectbox(f"Select GR for {uploaded_file.name}:", available_cols, index=get_default_index('GR', available_cols), key=f"gr_{i}")
                        col_nphi = st.selectbox(f"Select NPHI for {uploaded_file.name}:", available_cols, index=get_default_index('NPHI', available_cols), key=f"nphi_{i}")
                        col_rhob = st.selectbox(f"Select RHOB for {uploaded_file.name}:", available_cols, index=get_default_index('RHOB', available_cols), key=f"rhob_{i}")
                        col_rt = st.selectbox(f"Select RT for {uploaded_file.name}:", available_cols, index=get_default_index('RT', available_cols), key=f"rt_{i}")

                        mapping_dict = {col_gr: 'GR', col_nphi: 'NPHI', col_rhob: 'RHOB', col_rt: 'RT'}
                        df_clean = df.rename(columns=mapping_dict).copy()
                    else:
                        df_clean = df.copy()

                    df_clean = df_clean.dropna(subset=required_cols).copy()

                    # Convert NPHI from percentage to decimal if needed
                    if df_clean['NPHI'].max() > 1.0:
                        df_clean['NPHI'] = df_clean['NPHI'] / 100.0
                        st.info("ℹ️ NPHI curve detected in percentage units (%) and automatically converted to decimal fraction for the model.")

                    if st.button(f"Run Prediction & Plot — {uploaded_file.name}", key=f"btn_{i}"):
                        # Feature Engineering matching training pipeline
                        df_clean['Log_RT'] = np.log10(df_clean['RT'].clip(lower=0.001))

                        window = 5
                        df_clean['GR_rolling_mean'] = df_clean['GR'].rolling(window=window, center=True, min_periods=1).mean()
                        df_clean['NPHI_rolling_mean'] = df_clean['NPHI'].rolling(window=window, center=True, min_periods=1).mean()
                        df_clean['RHOB_rolling_mean'] = df_clean['RHOB'].rolling(window=window, center=True, min_periods=1).mean()
                        df_clean['Log_RT_rolling_mean'] = df_clean['Log_RT'].rolling(window=window, center=True, min_periods=1).mean()

                        # Extract exact 8 features in correct training order
                        features_df = df_clean[[
                            'GR', 'NPHI', 'RHOB', 'Log_RT', 
                            'GR_rolling_mean', 'NPHI_rolling_mean', 'RHOB_rolling_mean', 'Log_RT_rolling_mean'
                        ]].copy()

                        # Convert to NumPy array to bypass strict dataframe feature name checks
                        X_input = features_df.values

                        scaled_features = scaler.transform(X_input)
                        pred_log_k = model.predict(scaled_features)
                        df_clean['PRED_PERM_mD'] = 10 ** pred_log_k

                        st.subheader("Wireline Logs & Predicted Permeability Track")
                        fig, axes = plt.subplots(1, 4, figsize=(12, 8), sharey=True)
                        depth_col = df_clean.columns[0]

                        axes[0].plot(df_clean['GR'], df_clean[depth_col], color='green', lw=0.8)
                        axes[0].set_title('Gamma Ray\n(API)')
                        axes[0].set_xlabel('API')
                        axes[0].invert_yaxis()
                        axes[0].grid(True)

                        axes[1].plot(df_clean['RHOB'], df_clean[depth_col], color='red', lw=0.8)
                        axes[1].set_title('Bulk Density\n(g/cc)')
                        axes[1].set_xlabel('g/cc')
                        axes[1].grid(True)

                        axes[2].plot(df_clean['NPHI'], df_clean[depth_col], color='blue', lw=0.8)
                        axes[2].set_title('NPHI\n(v/v)')
                        axes[2].set_xlabel('v/v')
                        axes[2].set_xlim(0.45, -0.15)
                        axes[2].grid(True)

                        axes[3].plot(df_clean['PRED_PERM_mD'], df_clean[depth_col], color='purple', lw=0.8)
                        axes[3].set_title('Pred Permeability\n(mD)')
                        axes[3].set_xlabel('mD (Log)')
                        axes[3].set_xscale('log')
                        axes[3].grid(True)

                        axes[0].set_ylabel('Depth')
                        plt.tight_layout()
                        st.pyplot(fig)

                        csv_data = df_clean.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download Results ({uploaded_file.name})",
                            data=csv_data,
                            file_name=f"predicted_permeability_{uploaded_file.name}.csv",
                            mime="text/csv",
                            key=f"dl_{i}"
                        )

            except Exception as e:
                st.error(f"Error reading file {uploaded_file.name}: {e}")

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
