import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def full_preprocessing(data):
    # Step 1: Fill missing values
    data["DURATION"].fillna(data["DURATION"].median(), inplace=True)
    data["TIME_DIFF"].fillna(data["TIME_DIFF"].median(), inplace=True)
    data["WILAYA"].fillna("Unknown", inplace=True)
    data["PREV_WILAYA"].fillna("Unknown", inplace=True)
    data['EQUIPMENT_ID'].fillna('Other EQUI', inplace=True)

    # Step 2: Feature Engineering
    data["TIME_STAMP"] = pd.to_datetime(data["TIME_STAMP"], errors='coerce')
    data["OPTIONAL_FIELD_2"] = data["OPTIONAL_FIELD_2"].astype(str)

    # Extract Wilaya from OPTIONAL_FIELD_2
    def extract_wilaya(field):
        if pd.isna(field):
            return None
        matches = pd.Series([field]).str.findall(r'(\d{5})')[0]
        if not matches:
            return None
        wilayas = [m[1:3] for m in matches]
        if '00' in wilayas:
            wilayas = [m[:2] for m in matches]
        valid_wilayas = [w for w in wilayas if '01' <= w <= '48']
        return valid_wilayas[:2] if valid_wilayas else None

    data["WILAYA"] = data["OPTIONAL_FIELD_2"].apply(extract_wilaya)

    data.sort_values(["PHONE_NUMBER", "TIME_STAMP"], inplace=True)
    data["PREV_PHONE_NUMBER"] = data.groupby("PHONE_NUMBER")["PHONE_NUMBER"].shift(1)
    data["PREV_WILAYA"] = data.groupby("PHONE_NUMBER")["WILAYA"].shift(1)
    data["PREV_TIME"] = data.groupby("PHONE_NUMBER")["TIME_STAMP"].shift(1)
    data["TIME_DIFF"] = (data["TIME_STAMP"] - data["PREV_TIME"]).dt.total_seconds() / 60

    # SMS Features
    data["SMS_OUT_COUNT"] = data.groupby("CALLER_NUMBER")["CDR_SOURCE"].transform(lambda x: (x == "MSS SMSO").sum())
    unique_sms_dest = data[data["CDR_SOURCE"] == "MSS SMSO"].groupby("CALLER_NUMBER")["CALLED_NUMBER"].nunique().reset_index()
    unique_sms_dest.columns = ["CALLER_NUMBER", "UNIQUE_SMS_DEST"]
    data = data.merge(unique_sms_dest, on="CALLER_NUMBER", how="left").fillna({"UNIQUE_SMS_DEST": 0})

    sms_incoming = data[data["CDR_SOURCE"] == "MSS SMST"].groupby(["CALLER_NUMBER", "CALLED_NUMBER"]).size().reset_index(name="SMS_IN_COUNT")
    data = data.merge(sms_incoming, on=["CALLER_NUMBER", "CALLED_NUMBER"], how="left").fillna({"SMS_IN_COUNT": 0})

    # Step 3: Encoding
    data["EQUIPMENT_ID"] = data["EQUIPMENT_ID"].astype(str)
    data["EQUIPMENT_ID"] = LabelEncoder().fit_transform(data["EQUIPMENT_ID"])

    # Step 4: Scaling numeric features
    scaler = StandardScaler()
    num_cols = ['DURATION', 'TIME_DIFF', 'SMS_OUT_COUNT', 'UNIQUE_SMS_DEST', 'SMS_IN_COUNT', 'EQUIPMENT_ID']
    data[num_cols] = scaler.fit_transform(data[num_cols])

    # Step 5: One-hot encoding OPTIONAL_FIELD_2
    data = pd.get_dummies(data, columns=["OPTIONAL_FIELD_2"], drop_first=False)

    # Drop irrelevant columns
    data = data.drop(['TIME_STAMP', 'PREV_TIME', 'FILE_NAME', 'PREV_PHONE_NUMBER', 'Fraud'], axis=1, errors='ignore')

    return data
