import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import datetime, timedelta
from prophet import Prophet
import os

# Initialize Flask app
app = Flask(__name__)

# Define a directory to save models
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load or train the Prophet model
model_file = os.path.join(MODEL_DIR, 'prophet_bitcoin_model.pkl')

if os.path.exists(model_file):
    print(f"Loading existing model from {model_file}")
    with open(model_file, 'rb') as f:
        m = pickle.load(f)
else:
    print("Training new model...")
    # Load the dataset
    df = pd.read_csv('btc_usd_history.csv')
    df1 = df[['Date', 'Open']].rename(columns={'Date': 'ds', 'Open': 'y'})
    df1['ds'] = pd.to_datetime(df1['ds'])
    if df1['ds'].dt.tz is not None:
        df1['ds'] = df1['ds'].dt.tz_localize(None)
    df1 = df1.sort_values('ds')

    # Initialize and fit the model
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df1)

    # Save the trained model
    with open(model_file, 'wb') as f:
        pickle.dump(m, f)
    print(f"Model saved to {model_file}")

# Generate future dates starting from tomorrow (April 3, 2025) for 100 years (36,500 days)
start_date = datetime(2025, 4, 3)
future = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=36500, freq='D')})
forecast = m.predict(future)
# Ensure 'ds' in forecast is in string format for easier matching
forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')

# Route for the homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for the 404 page
@app.route('/404', methods=['GET'])
def not_found():
    return render_template('404.html')

# Route for the about page
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

# Route for the contact page
@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

# Route for the FAQ page
@app.route('/faq', methods=['GET'])
def faq():
    return render_template('faq.html')

# Route for the feature page
@app.route('/feature', methods=['GET'])
def feature():
    return render_template('feature.html')

# Route for the prediction page
@app.route('/Bitcoin', methods=['GET'])
def prediction():
    return render_template('predict.html')

# Route for handling prediction form submission (AJAX-compatible)
@app.route('/predict', methods=['POST'])
def y_predict():
    if request.method == "POST":
        # Get the date input from the form
        ds = request.form.get("Date")
        if not ds:
            return jsonify({'error': 'No date provided'}), 400

        print(f"Selected date: {ds}")

        try:
            # Convert selected date to datetime for comparison
            selected_date = datetime.strptime(ds, '%Y-%m-%d')
            # Generate 5 days of predictions (2 before, selected, 2 after)
            dates = [selected_date + timedelta(days=i) for i in range(-2, 3)]
            predictions = []

            for date in dates:
                date_str = date.strftime('%Y-%m-%d')
                # Find the prediction in the precomputed forecast
                pred_row = forecast[forecast['ds'] == date_str]
                if not pred_row.empty:
                    prediction = round(pred_row['yhat'].item(), 2)
                    predictions.append({
                        'date': date.strftime('%B %d, %Y'),
                        'price': f"{prediction:,.2f}",
                        'is_selected': date == selected_date
                    })
                else:
                    predictions.append({
                        'date': date.strftime('%B %d, %Y'),
                        'price': 'N/A',
                        'is_selected': date == selected_date
                    })

            return jsonify({'predictions': predictions})
        except ValueError as e:
            print(f"Error: {e}")
            return jsonify({'error': f"Invalid date or no prediction available for {ds}"}), 400

    return jsonify({'error': 'Method not allowed'}), 405

# Route for the roadmap page
@app.route('/roadmap', methods=['GET'])
def roadmap():
    return render_template('roadmap.html')

# Route for the service page
@app.route('/service', methods=['GET'])
def service():
    return render_template('service.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)  # Set debug=True for better error logging