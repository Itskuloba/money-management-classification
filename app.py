import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import datetime
import pdfplumber
import io
from rapidfuzz import fuzz, process  

# 1. Load artefacts (cached)
@st.cache_resource
def load_artifacts():
    model            = joblib.load('best_model_classifier')
    le               = joblib.load('label_encoder.joblib')
    merchant_counts  = joblib.load('merchant_counts.joblib')
    scaler           = joblib.load('scaler.joblib')
    imputation_vals  = joblib.load('imputation_values.joblib')
    expected_cols    = joblib.load('final_model_columns.joblib')
    return model, le, merchant_counts, scaler, imputation_vals, expected_cols

model, le, merchant_counts, scaler, imputation_values, expected_columns = load_artifacts()

# 2. Constants & defaults
DEFAULT_GENDER = 'Male'

data_keywords = ['tunukiwa','offers','bundles','airtime','cyber','safaricom','airtel','telkom']
bill_keywords = ['kplc','prepaid','postpaid','zuku','nairobi water','nhif','dstv']
bank_keywords = ['bank','equity','kcb','ncba','m-shwari']
supermarket_keywords = ['carrefour','quickmart','naivas','supermarket']
transport_keywords = ['uber','bolt','parking','go']
loan_keywords = ['loan','credit','mogo']
health_keywords = ['pharmacy','hospital','clinic']

pod_categories = ['morning','afternoon','evening','night']
gender_categories = ['Male','Female']

cols_to_scale = [
    'USER_AGE','USER_HOUSEHOLD','purchase_hour','purchase_day_of_week',
    'purchase_month','merchant_frequency','purchase_value_log','user_income_log'
]

# 3. CORRECTION DICTIONARY – Fixes known model mistakes
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

# 4. P2P & time helpers
def is_p2p(merchant_name):
    s = str(merchant_name).strip()
    business_indicators = [
        'supermarket','limited','ltd','plc','corporation','corp','company',
        'co','inc','enterprises','services','industries','hotel','restaurant',
        'bank','insurance','airlines','school','hospital','pharmacy',
        'mall','store','shop','market'
    ]
    s_low = s.lower()
    if any(ind in s_low for ind in business_indicators):
        return 0
    if s.isupper() or any(c.isdigit() for c in s):
        return 0
    if re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+){1,2}$', s):
        return 1
    return 0

def get_part_of_day(hour):
    if 5 <= hour < 12: return 'morning'
    elif 12 <= hour < 17: return 'afternoon'
    elif 17 <= hour < 21: return 'evening'
    else: return 'night'

# 5. Feature engineering – MERCHANT_NAME preserved only for rules
def engineer_features(input_df):
    df = input_df.copy()

    # Preserve MERCHANT_NAME
    if 'MERCHANT_NAME' in df.columns:
        merchant_name_series = df['MERCHANT_NAME'].copy()
    else:
        merchant_name_series = pd.Series([None] * len(df), name='MERCHANT_NAME')

    # Numeric cleaning
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

    # Categorical encoding
    df['part_of_day'] = pd.Categorical(df['part_of_day'], categories=pod_categories)
    df['USER_GENDER'] = pd.Categorical(df['USER_GENDER'], categories=gender_categories)
    df = pd.get_dummies(df, columns=['USER_GENDER', 'part_of_day'], drop_first=False)

    # Align to training columns
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]

    # Scaling
    df[cols_to_scale] = df[cols_to_scale].replace([np.inf, -np.inf], np.nan).fillna(0)
    df[cols_to_scale] = scaler.transform(df[cols_to_scale].astype(float))

    # Re-attach MERCHANT_NAME
    df['MERCHANT_NAME'] = merchant_name_series

    return df

# 6. M-Pesa parser
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

# 7. Download helpers
@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    out = io.BytesIO()
    df.to_excel(out, index=False, engine='openpyxl')
    return out.getvalue()

# 8. RULE ENGINE 
def apply_rules(df):
    if 'MERCHANT_NAME' not in df.columns:
        df['MERCHANT_NAME'] = None
    name_str = df['MERCHANT_NAME'].astype(str).str.strip()

    # 1. Exact corrections from dictionary
    for merchant, correct_cat in CORRECTION_RULES.items():
        mask = name_str.str.contains(re.escape(merchant), case=False, na=False)
        df.loc[mask, 'Predicted_Category'] = correct_cat

    # 2. Fuzzy-match fallback (90%+ similarity)
    merchants = list(CORRECTION_RULES.keys())
    for idx, name in enumerate(name_str):
        if pd.isna(name) or not name:
            continue
        match = process.extractOne(name, merchants, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 90:
            df.loc[idx, 'Predicted_Category'] = CORRECTION_RULES[match[0]]

    # 3. Keyword rules
    df.loc[name_str.str.contains('paybill', case=False, na=False), 'Predicted_Category'] = 'Bills & Fees'
    df.loc[name_str.str.contains('safaricom', case=False, na=False), 'Predicted_Category'] = 'Data & WiFi'
    df.loc[name_str.str.contains('naivas|carrefour|quickmart|supermarket', case=False, na=False), 'Predicted_Category'] = 'Groceries'

    # P2P
    df['is_p2p_tmp'] = df['MERCHANT_NAME'].apply(is_p2p)
    df.loc[df['is_p2p_tmp'] == 1, 'Predicted_Category'] = 'Family & Friends'
    df.drop('is_p2p_tmp', axis=1, inplace=True)

    # Airtime safety
    df.loc[name_str.str.contains('tunukiwa|bundles|offers|airtime', case=False, na=False), 'Predicted_Category'] = 'Data & WiFi'

    return df

# ----------------------------------------------------------------------
# 9. UI
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
            age      = st.number_input("Age", min_value=1, max_value=120, value=None)
            gender   = st.selectbox("Gender", ["Male","Female","Prefer not to say"], index=2)
            house    = st.number_input("Household Size", min_value=1, value=None)
            income   = st.number_input("Income (Ksh)", min_value=0, value=None)
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

            feats = engineer_features(df_in)
            model_input = feats[expected_columns]
            pred = le.inverse_transform(model.predict(model_input))[0]

            feats['Predicted_Category'] = pred
            final_df = apply_rules(feats)
            final_pred = final_df['Predicted_Category'].iloc[0]

            st.success(f"**{final_pred}**")

# ---------- M-Pesa Message ----------
elif method == "M-Pesa Message":
    st.header("Paste M-Pesa Message")
    msg = st.text_area("Message", height=120, placeholder="RKP... Confirmed. Ksh100 sent to...")
    c1, c2 = st.columns(2)
    with c1:
        age    = st.number_input("Age", min_value=1, value=None)
        gender = st.selectbox("Gender", ["Male","Female","Prefer not to say"])
    with c2:
        house  = st.number_input("Household", min_value=1, value=None)
        income = st.number_input("Income", min_value=0, value=None)
    if st.button("Classify"):
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
                    'PURCHASE_VALUE': parsed['PURCHASE_VALUE'],
                    'PURCHASED_AT': parsed['PURCHASED_AT'],
                    'USER_AGE': age,
                    'USER_GENDER': g,
                    'USER_HOUSEHOLD': house,
                    'USER_INCOME': income
                }])

                feats = engineer_features(df_in)
                model_input = feats[expected_columns]
                pred = le.inverse_transform(model.predict(model_input))[0]

                feats['Predicted_Category'] = pred
                final_df = apply_rules(feats)
                final_pred = final_df['Predicted_Category'].iloc[0]

                st.success(f"**{final_pred}**")

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
            # Read file
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

            # Original preview
            st.subheader("Original Statement (first 5 rows):")
            st.dataframe(df.head(5), use_container_width=True)

            # Map columns
            required = ['Completion Time', 'Details']
            if not all(col in df.columns for col in required):
                st.error(f"Missing columns: {set(required) - set(df.columns)}")
                st.stop()

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
                feats = engineer_features(input_df)

            # Prediction
            with st.spinner("Classifying..."):
                model_input = feats[expected_columns]
                raw_preds = le.inverse_transform(model.predict(model_input))
                df['Predicted_Category'] = raw_preds
                df = apply_rules(df)

            st.success("Classification Complete!")

            # Paginated table — NO COLORS, NO SOURCE
            show_cols = ['Completion Time','Details','Withdrawn','Paid In','Predicted_Category']
            show_cols = [c for c in show_cols if c in df.columns]

            page_size = 100
            total_pages = max(1, (len(df) - 1) // page_size + 1)
            page = st.slider("Page", 1, total_pages, 1, key="statement_page")
            start_idx = (page - 1) * page_size
            end_idx   = page * page_size

            st.dataframe(
                df[show_cols].iloc[start_idx:end_idx],
                use_container_width=True
            )
            st.caption(f"Showing rows {start_idx + 1}–{min(end_idx, len(df))} of {len(df)}")

            # Distribution
            st.subheader("Prediction Distribution")
            pred_dist = df['Predicted_Category'].value_counts().reset_index()
            pred_dist.columns = ['Category', 'Count']
            st.dataframe(pred_dist, use_container_width=True)

            # Download
            st.write("### Download your classified statement:")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download CSV", data=to_csv(df), file_name="classified.csv", mime="text/csv")
            with c2:
                st.download_button("Download Excel", data=to_excel(df), file_name="classified.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error: {e}")