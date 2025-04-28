from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('childhood_disease_model.pkl')
le = joblib.load('label_encoder.pkl')


data = pd.read_csv('Filtered_Childhood_Dataset.csv')
symptoms = data.columns[:-1].tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    selected_symptoms = []
    
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        if not selected_symptoms:
            result = {'disease': 'Please select at least one symptom', 'symptoms': []}
        else:
            # Create input
            input_data = pd.DataFrame(0, index=[0], columns=symptoms)
            for symptom in selected_symptoms:
                if symptom in input_data.columns:
                    input_data[symptom] = 1
            # Predict
            prediction = model.predict(input_data)
            disease = le.inverse_transform(prediction)[0]
            result = {'disease': disease, 'symptoms': selected_symptoms}
    
    return render_template('index.html', symptoms=symptoms, result=result)

if __name__ == '__main__':
    app.run(debug=True)