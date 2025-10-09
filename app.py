import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# --------------------------
# โหลด model + scaler + encoders
# --------------------------
model = load_model("loan_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
le_dict = pickle.load(open("le_dict.pkl", "rb"))
le_target = pickle.load(open("le_target.pkl", "rb"))

# --------------------------
# ตั้งค่า Theme
# --------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="centered",
)

# --------------------------
# CSS custom — โทนธนาคาร
# --------------------------
st.markdown("""
    <style>
        /* พื้นหลังหลัก */
        .main {
            background-color: #ffffff;
            font-family: "Segoe UI", sans-serif;
        }

        /* แถบด้านบน */
        .top-bar {
            background-color: #006d5b;
            color: white;
            padding: 18px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            border-radius: 0 0 15px 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }

        /* ปุ่ม */
        .stButton>button {
            background-color: #006d5b;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #008f75;
        }

        /* กล่องผลลัพธ์ */
        .result-box {
            text-align:center;
            border-radius:15px;
            padding:25px;
            margin-top:20px;
        }

        .approve {
            background-color:#e6ffee;
            border:2px solid #00b33c;
        }

        .reject {
            background-color:#ffe6e6;
            border:2px solid #ff3333;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# ส่วนหัวเว็บ
# --------------------------
st.markdown('<div class="top-bar">🏦 Loan Approval Prediction System</div>', unsafe_allow_html=True)

# คำอธิบายอยู่กลางจอ
st.markdown(
    """
    <div style="text-align: center; margin-top: 25px; margin-bottom: 5px;">
        <h5 style="font-size: 18px; max-width: 900px; margin: auto; color: #333;">
            กรอกข้อมูลสินเชื่อด้านล่างเพื่อดูผลการพิจารณาเบื้องต้นจากระบบ AI
        </h5>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------
# Input Form
# --------------------------
with st.form("loan_form"):
    # แถว 1
    cols1 = st.columns(6)
    Gender = cols1[0].selectbox("Gender", ["Male", "Female"])
    Married = cols1[1].selectbox("Married", ["Yes", "No"])
    Dependents = cols1[2].selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = cols1[3].selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = cols1[4].selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = cols1[5].number_input("Income", min_value=0)

    # แถว 2
    cols2 = st.columns(6)
    CoapplicantIncome = cols2[0].number_input("Co-Income", min_value=0)
    LoanAmount = cols2[1].number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = cols2[2].number_input("Term (Months)", min_value=0)
    Credit_History = cols2[3].selectbox("Credit History", [1.0, 0.0])
    Property_Area = cols2[4].selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    cols2[5].write("")  # เว้นว่างเพื่อให้ layout สวย

    submitted = st.form_submit_button("🔍 Predict Loan Status")

# --------------------------
# เมื่อกดปุ่ม
# --------------------------
if submitted:
    # สร้าง DataFrame
    input_dict = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }
    df_input = pd.DataFrame([input_dict])

    # แปลง categorical เป็นตัวเลข
    cat_cols = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
    for col in cat_cols:
        df_input[col] = le_dict[col].transform(df_input[col])

    # Scale features
    X_input_scaled = scaler.transform(df_input)

    # Predict
    pred_prob = model.predict(X_input_scaled)[0][0]
    pred_class = 1 if pred_prob >= 0.5 else 0
    pred_label = le_target.inverse_transform([pred_class])[0]  # 'Y' หรือ 'N'

    if pred_label == "Y":  # ผ่าน
        st.markdown(
            f"""
            <div style="text-align:center; background-color:#e6ffee; padding:25px; border-radius:15px; margin-top:20px;">
                <h2 style="color:#008000; font-size: 24px;">✅ ผลการพิจารณาเบื้องต้น: ผ่านการอนุมัติสินเชื่อ</h2>
                <p style="color:black;"><b>Probability:</b> {pred_prob:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:  # ไม่ผ่าน
        st.markdown(
            f"""
            <div style="text-align:center; background-color:#ffe6e6; padding:25px; border-radius:15px; margin-top:20px;">
                <h2 style="color:#cc0000; font-size: 24px;">❌ ผลการพิจารณาเบื้องต้น: ไม่ผ่านการอนุมัติสินเชื่อ</h2>
                <p style="color:black;">ระบบวิเคราะห์ว่าผู้กู้ <b>อาจไม่ผ่านการอนุมัติสินเชื่อ</b></p>
                <p style="color:black;"><b>Probability:</b> {pred_prob:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
