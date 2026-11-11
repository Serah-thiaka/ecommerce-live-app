from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained Random Forest model
model = joblib.load('ecommerce_behaviour_rf_model.pkl')
features = ['Age', 'Login_Frequency', 'Session_Duration_Avg', 'Pages_Per_Session', 'Cart_Abandonment_Rate', 'Total_Purchases']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    confidence_text = None
    show_popup = False

    if request.method == 'POST':
        # Get numbers directly from the web form
        age = int(request.form['age'])
        login_freq = int(request.form['login_freq'])
        session_time = float(request.form['session_time'])
        pages = float(request.form['pages'])
        cart_abandon = float(request.form['cart_abandon'])  # Kept as whole number (0-100) to match CSV
        purchases = int(request.form['purchases'])

        # Put into DataFrame and predict
        input_data = pd.DataFrame([[age, login_freq, session_time, pages, cart_abandon, purchases]], columns=features)
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            prediction_text = " LIKELY TO PURCHASE"
            confidence_text = f"Confidence: {prob:.2%}"
        else:
            prediction_text = "HESITANT VISITOR (HIGH DROP-OFF RISK)"
            confidence_text = f"Confidence: {1 - prob:.2%}"
            show_popup = True

    return render_template('index.html', prediction=prediction_text, confidence=confidence_text, popup=show_popup)

if __name__ == '__main__':
    app.run(debug=True)
