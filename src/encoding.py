from sklearn.preprocessing import LabelEncoder
import joblib
import os

ENCODER_PATH = "../models/label_encoder.joblib"

def fit_and_save_label_encoder(y):
    le = LabelEncoder()
    le.fit(y)
    os.makedirs("../models", exist_ok=True)
    joblib.dump(le, ENCODER_PATH)
    return le

def load_label_encoder():
    return joblib.load(ENCODER_PATH)