import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Loan Default Risk Predictor")
st.markdown("An XGBoost model to predict the likelihood of loan default. Built for credit risk assessment.")
st.divider()

st.title("Credit Default Risk scorer")
st.markdown("### Enter application details. Model explains decline reasons for SARB compliance.")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2721/2721622.png", width=100) # bank icon
    st.markdown("### About")
    st.write("This app uses an XGBoost model trained on loan data to predict default risk and explain the top factors using SHAP.")

st.markdown('#### Income')
INCOME1= st.number_input('Primary Annual income', min_value=0.01, max_value=50000000.00, step=5000.0)
INCOME2= st.number_input('Other annual income', min_value=0.00, max_value=1000000000.00, step=5000.0)

st.markdown('#### Expenditure')
EXPENDITURE1= st.slider('Amount spent on grossery', min_value=0.00, max_value=30000.00, step=10.00)
EXPENDITURE2= st.slider('Amount spent on Water and Electricity', min_value=0.00, max_value=30000.00, step=10.00)
EXPENDITURE3= st.slider('Amount spent on transport ', min_value=0.00, max_value=30000.00, step=10.0)
EXPENDITURE4= st.slider('Amount spent on Education', min_value=0.00, max_value=30000.00, step=10.0)
EXPENDITURE5= st.slider('Amount spent on airtime and data', min_value=0.00, max_value=10000.00, step=10.0)
EXPENDITURE6= st.slider('Amount spent on other loans', min_value=0.00, max_value=100000.00, step=10.0)
EXPENDITURE7= st.slider('Amount spent on Others eg intertainment', min_value=0.00, max_value=30000.00, step=10.0)

st.markdown('#### Investments')
INVESTMENT1= st.number_input('Amount invested in shares', min_value=0.00, max_value=1000000.00, step=5.0)
INVESTMENT2= st.number_input('Amount invested in retirement annuity', min_value=0.00, max_value=1000000.00, step=5.0)
INVESTMENT3= st.number_input('Amount invested in other investments', min_value=0.00, max_value=1000000.00, step=5.0)

st.markdown('#### Credit ')
AMT_CREDIT=st.number_input('Credit_Amount',min_value=0.00, max_value=100000000.00, step=10.0)

st.markdown('#### Bureau Status')
BUREAU_DAYS_CREDIT_MIN=st.slider('BUREAU_DAYS_CREDIT_MIN',min_value=-3000, max_value=0, step=1)
BUREAU_DAYS_CREDIT_MAX=st.slider('BUREAU_DAYS_CREDIT_MAX',min_value=-3000, max_value=0, step=1)
BUREAU_CREDIT_ACTIVE=st.slider('BUREAU CREDIT ACTIVE STATUS',min_value=0, max_value=50, step=1)
TOTAL_BUREAU_CREDIT_DAY_OVERDUE=st.number_input('TOTAL_BUREAU_CREDIT_DAY_OVERDUE',min_value=0, max_value=1050, step=1)

st.markdown('#### Previous Application History')
NUMBER_OF_PAST_APPS=st.radio('NUMBER_OF_PAST_APP',options=[1,2,3,4,5,6,7,8,9,10],index=0)
PREVIOUS_REFUSED_RATIO=st.number_input('PREVIOUS_REFUSED_RATIO',min_value=0.00, max_value=1.00, step=0.01)

st.markdown('#### Applicant Age and Employment period')
AGE=st.number_input('AGE',min_value=18, max_value=75, step=1)
YEARS_EMPLOYED=st.slider("YEARS_EMPLOYED",min_value=0, max_value=45, step=1)

st.markdown('#### EXT_SOURCE SCORES')

EXT_SOURCE_1=st.number_input('EXT_SOURCE_1',min_value=0.00, max_value=1.0, step=0.01)
EXT_SOURCE_2=st.number_input('EXT_SOURCE_2',min_value=0.00, max_value=1.0, step=0.01)
EXT_SOURCE_3=st.number_input('EXT_SOURCE_3',min_value=0.00, max_value=1.0, step=0.01)

#CODE_GENDER=st.radio('Select Your gender : Female=0;  Male=1 ;  XNA=2',options=[0,1,2],index=0)
st.markdown('#### Number of children')
CNT_CHILDREN=st.selectbox('NUMBER OF CHILDREN',options=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],index=0)

AMT_INCOME_TOTAL=(INCOME1+INCOME2)

ANNUAL_INVESTMENTS_AMT=(INVESTMENT1+INVESTMENT2+INVESTMENT3)*12
ANNUAL_TOTAL_EXPENDITURE=(EXPENDITURE1+EXPENDITURE2+EXPENDITURE3+EXPENDITURE4+EXPENDITURE5+EXPENDITURE6+EXPENDITURE7)*12
AMT_ANNUITY=(ANNUAL_INVESTMENTS_AMT+ANNUAL_TOTAL_EXPENDITURE)

DTI_RATIO=AMT_ANNUITY/AMT_INCOME_TOTAL
CREDIT_TO_INCOME_RATIO=AMT_CREDIT/AMT_INCOME_TOTAL

EXT_SOURCE_MEAN=(EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3)/3

@st.cache_resource
def load_models():
    calibrated_model = joblib.load('src/XGB_Calibrated_Model.pkl') # for prediction
    base_model = joblib.load('src/XGB_Base_Model.pkl')             # for SHAP
    feature_names = joblib.load('src/feature_names.pkl')
    return calibrated_model, base_model, feature_names

calibrated_model, base_model, feature_names = load_models()



X_input=pd.DataFrame([[DTI_RATIO,CREDIT_TO_INCOME_RATIO, BUREAU_DAYS_CREDIT_MIN, BUREAU_DAYS_CREDIT_MAX,
          BUREAU_CREDIT_ACTIVE, TOTAL_BUREAU_CREDIT_DAY_OVERDUE, NUMBER_OF_PAST_APPS,
          PREVIOUS_REFUSED_RATIO, YEARS_EMPLOYED, EXT_SOURCE_MEAN, AGE,CNT_CHILDREN]],columns=feature_names)

# Prediction
prediction = calibrated_model.predict(X_input)
prediction_proba = calibrated_model.predict_proba(X_input)[:, 1]
st.write(f"**Prediction:** {'Default Risk' if prediction[0] == 1 else 'No Default Risk'}")
st.write(f"**Probability of Default:** {prediction_proba[0]:.2%}")

# SHAP Explanation - With a base model for SHAP values


st.subheader("Feature Impact - SHAP Waterfall")

try:
    # 1. CREATE THE EXPLAINER FIRST
    explainer = shap.TreeExplainer(base_model, feature_perturbation="interventional")
    
    # 2. GET SHAP VALUES
    shap_values = explainer.shap_values(X_input)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    # 3. PLOT
    plt.figure(figsize=(10,6))
    exp = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=X_input.iloc[0],
        feature_names=feature_names)
    
    shap.waterfall_plot(exp, max_display=10, show=False)

    # Fix the +0 label issue
    ax = plt.gca()
    for t in ax.texts:
        txt = t.get_text()
        if txt.startswith("+") or txt.startswith("-"):
            try:
                val = float(txt.replace("+","").replace("-",""))
                t.set_text(f"{val:.3f}")
            except: pass

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()
    
except Exception as e:
    st.error(f"SHAP Error: {e}")

#except Exception as e:
#   st.warning(f"SHAP plot skipped: {e}")
#plt.clf()

if st.button("Score"):
    Threshold=0.32
    prob=calibrated_model.predict_proba([[DTI_RATIO,CREDIT_TO_INCOME_RATIO, BUREAU_DAYS_CREDIT_MIN, BUREAU_DAYS_CREDIT_MAX,
          BUREAU_CREDIT_ACTIVE, TOTAL_BUREAU_CREDIT_DAY_OVERDUE, NUMBER_OF_PAST_APPS,
          PREVIOUS_REFUSED_RATIO, YEARS_EMPLOYED, EXT_SOURCE_MEAN, AGE,CNT_CHILDREN]])[0,1]
    st.metric(
        label="Probability of Default",
        value=f"{prob:.2%}",
        delta="-Low Risk" if prob < Threshold  else "High Risk",
        delta_color="inverse" if prob < Threshold else "normal")

   
     if prob<Threshold:
        st.success("The loan is approved")
    elif prob<0.4:
        st.success("Low risk of default. Credit profile looks good. Manual verification is reccomended")
        
        with st.expander("**Top reasons for score**"):
            st.write("1. Low exteranal source mean score")
            st.write("2. High DEBT TO INCOME RATIO")
            st.write("3. Low bureau score")
            st.write("4. Unstable employment")
    else:
        st.error("High risk of default. Consider improving credit profile.")


if st.button("Predict Risk"):
    X_input = pd.DataFrame([input_data], columns=feature_names)
    
    prediction = calibrated_model.predict(X_input)
    prob = calibrated_model.predict_proba(X_input)[:, 1][0]
    
    st.subheader("Probability of Default")
    st.metric(label="Score", value=f"{prob:.2%}")
    
    if prob < 0.32:
        st.error(f"High Risk: {prob:.2%} chance of default")
        decision = "APPROVED"
    elif prob < 0.4 :
        st.success(f"Low Risk: {prob:.2%} chance of default")
        decision = "Credit profile looks better. Manual verification is reccomended"
    else:
        st.success(f"Low Risk: {prob:.2%} chance of default")
        decision = "REJECTED"
    
    st.write(f"**The loan is {decision}**")

    # SHAP plot code here...

    # --- DOWNLOAD BUTTON - MUST BE INSIDE THE IF BLOCK ---
    report_data = {
        "Feature": list(input_data.keys()),
        "Value": list(input_data.values())
    }
    report_df = pd.DataFrame(report_data)

    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Report as CSV",
        data=csv,
        file_name=f"loan_prediction_{decision}_{prob:.0%}.csv",
        mime='text/csv'
    )
        
