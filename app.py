import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import datetime
import pdfplumber
import io
from rapidfuzz import fuzz, process  
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# DEBUG: Show versions 
# ----------------------------------------------------------------------
import xgboost as xgb
st.sidebar.success(f"XGBoost {xgb.__version__} – Python {__import__('sys').version.split()[0]}")

# ----------------------------------------------------------------------
# Load artefacts 
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model           = joblib.load('best_model_classifier')
    le              = joblib.load('label_encoder.joblib')
    merchant_counts = joblib.load('merchant_counts.joblib')
    scaler          = joblib.load('scaler.joblib')
    imputation_vals = joblib.load('imputation_values.joblib')
    expected_cols   = joblib.load('final_model_columns.joblib')
    return model, le, merchant_counts, scaler, imputation_vals, expected_cols

# Load artifacts
try:
    model, le, merchant_counts, scaler, imputation_values, expected_columns = load_artifacts()
    st.success("Model and artifacts loaded successfully!")
except Exception as e:
    st.error(f"Critical error during initialization: {str(e)}")
    st.stop()

# ----------------------------------------------------------------------
# Constants & defaults
# ----------------------------------------------------------------------
DEFAULT_GENDER = 'Male'

data_keywords = ['tunukiwa','offers','bundles','airtime','cyber','safaricom','airtel','telkom']
bill_keywords = ['kplc','prepaid','postpaid','zuku','nairobi water','nhif','dstv']
bank_keywords = ['bank','equity','kcb','ncba','m-shwari']
supermarket_keywords = ['carrefour','quickmart','naivas','supermarket','clean shelf']
transport_keywords = ['uber','bolt','parking','go']
loan_keywords = ['loan','credit','mogo']
health_keywords = ['pharmacy','hospital','clinic']

pod_categories = ['morning','afternoon','evening','night']
gender_categories = ['Male','Female']

cols_to_scale = [
    'USER_AGE','USER_HOUSEHOLD','purchase_hour','purchase_day_of_week',
    'purchase_month','merchant_frequency','purchase_value_log','user_income_log'
]

# ----------------------------------------------------------------------
# CORRECTION DICTIONARY
# ----------------------------------------------------------------------
CORRECTION_RULES = {
    "GLADWELL MBURU":               "Miscellaneous",
    "HARRISON JUMA AYUAK":          "Going out",
    "RUBIS ENJOY UN AVENUE":        "Going out",
    "BRIOCHE RUBIS LANGATA":        "Going out",
    "APS ABC PARKING":              "Transport & Fuel",
    "CAPTON ENTERPRISES NRBI WEST": "Transport & Fuel",
    "ROBERT KARIUKI":               "Bills & Fees",
    "CHRISTINE OOKO":               "Miscellaneous",
    "NAIROBI JAVA HOUSE SARIT CENTRE": "Going out",
    "SAFARICOM LIMITED":            "Data & WiFi",
    "JULLY AKIN":                   "Groceries",
    "FARMERS BUTCHERY":             "Shopping",
    "METROMART LTD":                "Shopping",
    "LILIAN GICHU":                 "Family & Friends",
    "VENDIT LIMITED":               "Bills & Fees",
    "NHIF":                         "Bills & Fees",
    "DR CECILIA":                   "Miscellaneous",
    "COLLINS OUMA":                 "Family & Friends",
    "MABENAN CYBER":                "Bills & Fees",
    "PESAPAL FOR":                  "Bills & Fees",
    "ZILLIONS CREDIT LIMITED":      "Bills & Fees",
    "MICHAEL MUSEMBI":              "Miscellaneous",
    "NEBERT GITAU":                 "Transport & Fuel",
    "JOSEPHINE THUKU":              "Miscellaneous",
    "OUTSKIRTS PLACE":              "Transport & Fuel",
    "SIXTUS ABOLALA":               "Transport & Fuel",
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def get_part_of_day(hour):
    if 5 <= hour < 12: return 'morning'
    elif 12 <= hour < 17: return 'afternoon'
    elif 17 <= hour < 21: return 'evening'
    else: return 'night'

# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def engineer_features(input_df):
    df = input_df.copy()

    # PRESERVE ORIGINAL COLUMNS
    purchase_value_series = df['PURCHASE_VALUE'].copy() if 'PURCHASE_VALUE' in df.columns else pd.Series([0.0] * len(df))
    merchant_name_series = df['MERCHANT_NAME'].copy() if 'MERCHANT_NAME' in df.columns else pd.Series([None] * len(df), name='MERCHANT_NAME')
    purchased_at_series = df['PURCHASED_AT'].copy() if 'PURCHASED_AT' in df.columns else pd.Series([pd.Timestamp.now()] * len(df))

    # Clean numeric
    for col in ['USER_AGE', 'USER_HOUSEHOLD', 'USER_INCOME', 'PURCHASE_VALUE']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].clip(lower=0)
            if col in imputation_values:
                cap = imputation_values.get(col, 0) * 10
                df[col] = df[col].clip(upper=cap)

    for col in ['USER_AGE', 'USER_HOUSEHOLD', 'USER_INCOME']:
        if col in df.columns and col in imputation_values:
            df[col] = df[col].fillna(imputation_values[col])

    if 'USER_GENDER' in df.columns:
        df['USER_GENDER'] = df['USER_GENDER'].fillna(DEFAULT_GENDER)

    # Time features
    df['PURCHASED_AT'] = pd.to_datetime(df['PURCHASED_AT'], errors='coerce').fillna(pd.Timestamp("2024-01-01"))
    df['purchase_hour']        = df['PURCHASED_AT'].dt.hour
    df['purchase_day_of_week'] = df['PURCHASED_AT'].dt.dayofweek
    df['purchase_month']       = df['PURCHASED_AT'].dt.month
    df['is_weekend']           = (df['purchase_day_of_week'] >= 5).astype(int)
    df['part_of_day']          = df['purchase_hour'].apply(get_part_of_day)

    # Keyword flags
    df['merchant_name_clean'] = df['MERCHANT_NAME'].astype(str).str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    df['merchant_norm'] = df['MERCHANT_NAME'].astype(str).str.upper().str.replace(r'[^A-Z0-9 ]', '', regex=True).str.strip()

    df['is_data_purchase'] = df['merchant_name_clean'].apply(lambda x: any(k in x for k in data_keywords)).astype(int)
    df['is_bill_payment']  = df['merchant_name_clean'].apply(lambda x: any(k in x for k in bill_keywords)).astype(int)
    df['is_bank']          = df['merchant_name_clean'].apply(lambda x: any(k in x for k in bank_keywords)).astype(int)
    df['is_supermarket']   = df['merchant_name_clean'].apply(lambda x: any(k in x for k in supermarket_keywords)).astype(int)
    df['is_transport']     = df['merchant_name_clean'].apply(lambda x: any(k in x for k in transport_keywords)).astype(int)
    df['is_loan']          = df['merchant_name_clean'].apply(lambda x: any(k in x for k in loan_keywords)).astype(int)
    df['is_health']        = df['merchant_name_clean'].apply(lambda x: any(k in x for k in health_keywords)).astype(int)

    df['merchant_frequency'] = df['merchant_norm'].map(merchant_counts).fillna(0)

    # Log transforms
    df['PURCHASE_VALUE'] = df['PURCHASE_VALUE'].fillna(0).clip(lower=0)
    df['purchase_value_log'] = np.log1p(df['PURCHASE_VALUE'])
    df['USER_INCOME'] = df['USER_INCOME'].fillna(0).clip(lower=0)
    df['user_income_log'] = np.log1p(df['USER_INCOME'])

    # Categorical
    df['part_of_day'] = pd.Categorical(df['part_of_day'], categories=pod_categories)
    df['USER_GENDER'] = pd.Categorical(df['USER_GENDER'], categories=gender_categories)
    df = pd.get_dummies(df, columns=['USER_GENDER', 'part_of_day'], drop_first=False)

    # ALIGN TO MODEL COLUMNS
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    model_df = df[expected_columns].copy()

    # SCALE
    model_df[cols_to_scale] = model_df[cols_to_scale].replace([np.inf, -np.inf], np.nan).fillna(0)
    model_df[cols_to_scale] = scaler.transform(model_df[cols_to_scale].astype(float))

    # RE-ATTACH ORIGINAL COLUMNS
    df['MERCHANT_NAME'] = merchant_name_series
    df['PURCHASE_VALUE'] = purchase_value_series
    df['PURCHASED_AT'] = purchased_at_series

    return df, model_df  # full_df, model_input

# ----------------------------------------------------------------------
# M-Pesa parser
# ----------------------------------------------------------------------
def parse_mpesa_message(msg):
    data = {'MERCHANT_NAME': None, 'PURCHASE_VALUE': None, 'PURCHASED_AT': None}
    amt = re.search(r'Ksh([\d,\.]+)', msg)
    if amt: data['PURCHASE_VALUE'] = float(amt.group(1).replace(',',''))
    merchant = re.search(r'to\s+([A-Z0-9\s\.\-]+?)\s+(?:on|Till)', msg, re.I)
    if merchant: data['MERCHANT_NAME'] = merchant.group(1).strip()
    date = re.search(r'on\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+at\s+(\d{1,2}:\d{2}\s*[AP]M)', msg, re.I)
    if date:
        dt_str = f"{date.group(1)} {date.group(2)}"
        try:
            data['PURCHASED_AT'] = pd.to_datetime(dt_str, format="%d/%m/%y %I:%M %p", errors='coerce')
        except:
            data['PURCHASED_AT'] = pd.to_datetime(dt_str, format="%d/%m/%Y %I:%M %p", errors='coerce')
    if not data['PURCHASED_AT']: data['PURCHASED_AT'] = pd.Timestamp.now()
    return data

# ----------------------------------------------------------------------
# Download helpers
# ----------------------------------------------------------------------
@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    out = io.BytesIO()
    df.to_excel(out, index=False, engine='openpyxl')
    return out.getvalue()

# ----------------------------------------------------------------------
# RULE ENGINE
# ----------------------------------------------------------------------
def apply_rules(df):
    df = df.copy()
    
    if 'MERCHANT_NAME' not in df.columns:
        df['MERCHANT_NAME'] = df.get('Details', '')
    df['MERCHANT_NAME'] = df['MERCHANT_NAME'].astype(str).fillna('').str.strip()
    name_clean = df['MERCHANT_NAME'].str.lower()
    name_str = df['MERCHANT_NAME']

    if 'Predicted_Category' not in df.columns:
        df['Predicted_Category'] = 'Miscellaneous'

    # Exact matches
    for merchant, cat in CORRECTION_RULES.items():
        mask = name_str.str.contains(re.escape(merchant), case=False, na=False)
        df.loc[mask, 'Predicted_Category'] = cat

    # Fuzzy
    merchants = list(CORRECTION_RULES.keys())
    for idx, name in enumerate(name_str):
        if not name or pd.isna(name):
            continue
        match = process.extractOne(name, merchants, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 90:
            df.loc[idx, 'Predicted_Category'] = CORRECTION_RULES[match[0]]

    # Core rules
    df.loc[name_clean.str.contains('paybill|safaricom|airtime|bundles|tunukiwa|cyber', na=False), 'Predicted_Category'] = 'Data & WiFi'
    df.loc[name_clean.str.contains('naivas|quick mart|clean shelf|carrefour|supermarket', na=False), 'Predicted_Category'] = 'Groceries'
    df.loc[name_clean.str.contains('pay bill|e-citizen|kcb paybill|nhif|kplc|dstv|zuku', na=False), 'Predicted_Category'] = 'Bills & Fees'

    # P2P
    def is_person_name(name):
        s = str(name)
        if re.search(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}\b', s):
            bad = ['ltd','limited','bank','shop','mall','hotel','restaurant','cyber','paybill','totalenergies']
            if any(b in s.lower() for b in bad):
                return False
            return True
        return False
    df.loc[df['MERCHANT_NAME'].apply(is_person_name), 'Predicted_Category'] = 'Family & Friends'

    # Small business
    df.loc[name_clean.str.contains('small business', na=False), 'Predicted_Category'] = 'Family & Friends'

    # Withdrawals
    df.loc[name_clean.str.contains('withdraw|m-shwari|kcb m-pesa withdraw|atm', na=False), 'Predicted_Category'] = 'Miscellaneous'

    # Restaurants
    df.loc[name_clean.str.contains('java|galitos|waterfront|gardens|alchemist |bar|club|eat|cafe|restaurant|bistro', na=False), 'Predicted_Category'] = 'Going out'

    # Fuel
    df.loc[name_clean.str.contains('totalenergies|rubis|shell|oil', na=False), 'Predicted_Category'] = 'Transport & Fuel'

    # NEW FIXES
    df.loc[name_clean.str.contains('transfer of funds charge|pay merchant charge', na=False), 'Predicted_Category'] = 'Bills & Fees'
    df.loc[name_clean.str.contains('airtime|bundle|data|safaricom data', na=False), 'Predicted_Category'] = 'Data & WiFi'
    df.loc[name_clean.str.contains('loan request|loan repayment|loan disburse|m-shwari loan', na=False), 'Predicted_Category'] = 'Bills & Fees'
    df.loc[name_clean.str.contains('pay bill|e-citizen|paybill', na=False), 'Predicted_Category'] = 'Bills & Fees'
    df.loc[name_clean.str.contains('cyber|communications|yathui|sefran', na=False), 'Predicted_Category'] = 'Data & WiFi'

    return df

# ----------------------------------------------------------------------
# Safe Prediction Function
# ----------------------------------------------------------------------
def safe_predict(model, model_input):
    try:
        predictions = model.predict(model_input)
        return le.inverse_transform(predictions)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return ['Miscellaneous'] * len(model_input)

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Money Manager", layout="wide")
st.title("Money Management Classifier")
st.write("Classify your M-Pesa transactions instantly.")

method = st.sidebar.radio("Input Method", ["Manual", "M-Pesa Message", "Upload Statement"])

# ---------- Manual ----------
if method == "Manual":
    st.header("Enter Transaction")
    with st.form("manual"):
        c1, c2 = st.columns(2)
        with c1:
            merchant = st.text_input("Merchant Name*", placeholder="Naivas, SAFARICOM")
            amount   = st.number_input("Amount (Ksh)*", min_value=0.0, step=10.0)
            date     = st.date_input("Date*", datetime.date.today())
            time     = st.time_input("Time*", datetime.datetime.now().time())
        with c2:
            age      = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender   = st.selectbox("Gender", ["Male","Female","Prefer not to say"], index=2)
            house    = st.number_input("Household Size", min_value=1, value=3)
            income   = st.number_input("Income (Ksh)", min_value=0, value=50000)
        submitted = st.form_submit_button("Classify")

    if submitted:
        if not merchant or amount <= 0:
            st.error("Merchant name & amount required.")
        else:
            g = None if gender == "Prefer not to say" else gender
            dt = pd.Timestamp.combine(date, time)
            df_in = pd.DataFrame([{
                'MERCHANT_NAME': merchant,
                'PURCHASE_VALUE': amount,
                'PURCHASED_AT': dt,
                'USER_AGE': age,
                'USER_GENDER': g,
                'USER_HOUSEHOLD': house,
                'USER_INCOME': income
            }])

            try:
                full_df, model_df = engineer_features(df_in)
                pred = safe_predict(model, model_df)
                full_df['Predicted_Category'] = pred[0] if len(pred) > 0 else 'Miscellaneous'
                final_df = apply_rules(full_df)
                final_pred = final_df['Predicted_Category'].iloc[0]

                st.success(f"**Predicted Category: {final_pred}**")
                
                with st.expander("Transaction Details"):
                    st.write(f"**Merchant:** {merchant}")
                    st.write(f"**Amount:** Ksh {amount:,.2f}")
                    st.write(f"**Date/Time:** {dt.strftime('%Y-%m-%d %H:%M')}")
                    
            except Exception as e:
                st.error(f"Classification error: {e}")

# ---------- M-Pesa Message ----------
elif method == "M-Pesa Message":
    st.header("Paste M-Pesa Message")
    msg = st.text_area("Message", height=120, placeholder="RKP... Confirmed. Ksh100 sent to...")
    c1, c2 = st.columns(2)
    with c1:
        age    = st.number_input("Age", min_value=1, value=30, key="msg_age")
        gender = st.selectbox("Gender", ["Male","Female","Prefer not to say"], key="msg_gender")
    with c2:
        house  = st.number_input("Household", min_value=1, value=3, key="msg_house")
        income = st.number_input("Income", min_value=0, value=50000, key="msg_income")
    
    if st.button("Classify Message"):
        if not msg:
            st.error("Paste a message.")
        else:
            parsed = parse_mpesa_message(msg)
            if not parsed['MERCHANT_NAME']:
                st.error("Could not extract merchant/amount.")
            else:
                g = None if gender == "Prefer not to say" else gender
                df_in = pd.DataFrame([{
                    'MERCHANT_NAME': parsed['MERCHANT_NAME'],
                    'PURCHASE_VALUE': parsed['PURCHASE_VALUE'] or 0.0,
                    'PURCHASED_AT': parsed['PURCHASED_AT'],
                    'USER_AGE': age,
                    'USER_GENDER': g,
                    'USER_HOUSEHOLD': house,
                    'USER_INCOME': income
                }])

                try:
                    full_df, model_df = engineer_features(df_in)
                    pred = safe_predict(model, model_df)
                    full_df['Predicted_Category'] = pred[0] if len(pred) > 0 else 'Miscellaneous'
                    final_df = apply_rules(full_df)
                    final_pred = final_df['Predicted_Category'].iloc[0]

                    st.success(f"**Predicted Category: {final_pred}**")
                    
                    with st.expander("Parsed Details"):
                        st.write(f"**Merchant:** {parsed['MERCHANT_NAME']}")
                        st.write(f"**Amount:** Ksh {parsed['PURCHASE_VALUE'] or 0:,.2f}")
                        st.write(f"**Date/Time:** {parsed['PURCHASED_AT']}")
                        
                except Exception as e:
                    st.error(f"Classification error: {e}")

# ---------- Upload Statement ----------
else:
    st.header("Upload M-Pesa Statement")
    file = st.file_uploader("CSV / Excel / PDF", type=["csv","xlsx","pdf"])
    if file and file.name.endswith('.pdf'):
        st.text_input("PDF Password (if any)", type="password", key="pdf_pwd")

    c1, c2 = st.columns(2)
    with c1:
        age    = st.number_input("Age", min_value=1, value=25, key="batch_age")
        gender = st.selectbox("Gender", ["Male","Female"], key="batch_gender")
    with c2:
        house  = st.number_input("Household", min_value=1, value=3, key="batch_house")
        income = st.number_input("Income", min_value=0, value=90000, key="batch_income")

    if file and st.button("Classify Statement"):
        try:
            # Load file
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                pwd = st.session_state.get("pdf_pwd", "")
                with pdfplumber.open(file, password=pwd) as pdf:
                    tables = [p.extract_table() for p in pdf.pages if p.extract_table()]
                if not tables:
                    st.error("No tables found in PDF.")
                    st.stop()
                df = pd.concat([pd.DataFrame(t[1:], columns=t[0]) for t in tables if t], ignore_index=True)

            df.columns = [str(c).replace('\n', ' ') for c in df.columns]

            st.subheader("Original Statement (first 5 rows):")
            st.dataframe(df.head(5), use_container_width=True)

            required = ['Completion Time', 'Details']
            if not all(col in df.columns for col in required):
                st.error(f"Missing columns: {set(required) - set(df.columns)}")
                st.stop()

            # Build input
            input_df = pd.DataFrame()
            input_df['PURCHASED_AT']   = pd.to_datetime(df['Completion Time'])
            input_df['MERCHANT_NAME']  = df['Details']
            withdrawn = pd.to_numeric(df.get('Withdrawn'), errors='coerce').fillna(0)
            paid_in   = pd.to_numeric(df.get('Paid In'), errors='coerce').fillna(0)
            input_df['PURCHASE_VALUE'] = withdrawn.mask(withdrawn == 0, paid_in)
            input_df['USER_AGE']       = age
            input_df['USER_GENDER']    = gender
            input_df['USER_HOUSEHOLD'] = house
            input_df['USER_INCOME']    = income

            # Engineering
            with st.spinner("Engineering features..."):
                full_df, model_df = engineer_features(input_df)

            # Prediction
            with st.spinner("Classifying transactions..."):
                raw_preds = safe_predict(model, model_df)
                full_df['Predicted_Category'] = raw_preds
                df = full_df.copy()
                df = apply_rules(df)

            st.success("Classification Complete!")

            # SPENDING CHARTS
            st.markdown("---")
            st.subheader("Spending Analysis")

            spend_df = df[df['PURCHASE_VALUE'] < 0].copy()
            spend_df['Amount'] = spend_df['PURCHASE_VALUE'].abs()

            if len(spend_df) > 0:
                cat_spend = spend_df.groupby('Predicted_Category')['Amount'].sum().sort_values(ascending=False)
                fig_bar = px.bar(cat_spend.reset_index(), x='Predicted_Category', y='Amount',
                                 title="Total Spending by Category", color='Predicted_Category',
                                 text='Amount')
                fig_bar.update_traces(texttemplate='Ksh%{text:,.0f}', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)

                fig_pie = px.pie(cat_spend.reset_index(), values='Amount', names='Predicted_Category',
                                 title="Spending Distribution", hole=0.4)
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

                daily = spend_df.groupby(spend_df['PURCHASED_AT'].dt.date)['Amount'].sum().tail(30)
                fig_line = px.line(daily.reset_index(), x='PURCHASED_AT', y='Amount',
                                   title="Daily Spending (Last 30 Days)")
                fig_line.update_traces(line=dict(width=3))
                st.plotly_chart(fig_line, use_container_width=True)

                top_merchants = spend_df.groupby('MERCHANT_NAME')['Amount'].sum().sort_values(ascending=False).head(5)
                st.write("**Top 5 Merchants by Spend**")
                st.dataframe(top_merchants.apply(lambda x: f"Ksh {x:,.0f}").reset_index(), use_container_width=True)
            else:
                st.info("No spending transactions found to analyze.")

            # Table + Download
            show_cols = ['Completion Time','Details','Withdrawn','Paid In','Predicted_Category']
            show_cols = [c for c in show_cols if c in df.columns]

            page_size = 100
            total_pages = max(1, (len(df) - 1) // page_size + 1)
            page = st.slider("Page", 1, total_pages, 1, key="statement_page")
            start_idx = (page - 1) * page_size
            end_idx   = page * page_size

            st.dataframe(df[show_cols].iloc[start_idx:end_idx], use_container_width=True)
            st.caption(f"Showing rows {start_idx + 1}–{min(end_idx, len(df))} of {len(df)}")

            st.subheader("Prediction Distribution")
            pred_dist = df['Predicted_Category'].value_counts().reset_index()
            pred_dist.columns = ['Category', 'Count']
            st.dataframe(pred_dist, use_container_width=True)

            st.write("### Download your classified statement:")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download CSV", data=to_csv(df), file_name="classified.csv", mime="text/csv")
            with c2:
                st.download_button("Download Excel", data=to_excel(df), file_name="classified.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error processing statement: {e}")