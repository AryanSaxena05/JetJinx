from flask import Flask, render_template, request, jsonify
from predict_delays import predict_delay
import logging
import joblib
from insights_utils import get_monthly_avg_delay
from optimization_utils import get_optimization_result

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Log the incoming form data
            logger.debug(f"Received form data: {request.form}")
            
            # Validate required fields
            required_fields = ['month', 'day_of_week', 'airline', 'origin', 
                             'destination', 'scheduled_departure', 'distance']
            
            for field in required_fields:
                if field not in request.form:
                    raise ValueError(f"Missing required field: {field}")
            
            # Convert and validate numeric fields
            try:
                data = {
                    'MONTH': int(request.form['month']),
                    'DAY_OF_WEEK': int(request.form['day_of_week']),
                    'AIRLINE': str(request.form['airline']).upper(),
                    'ORIGIN_AIRPORT': str(request.form['origin']).upper(),
                    'DESTINATION_AIRPORT': str(request.form['destination']).upper(),
                    'SCHEDULED_DEPARTURE': int(request.form['scheduled_departure']),
                    'DISTANCE': float(request.form['distance'])
                }
            except ValueError as e:
                raise ValueError(f"Invalid numeric value: {str(e)}")
            
            # Log the processed data
            logger.debug(f"Processed data: {data}")
            
            # Make prediction
            prediction, probability = predict_delay(data)
            
            return jsonify({
                'status': 'Delayed' if prediction == 1 else 'Not Delayed',
                'probability': f"{probability:.2%}"
            })
            
        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}")
            return jsonify({'error': str(e)}), 400
            
    return render_template('predict.html')

@app.route('/insights/monthly-delay-data')
def monthly_delay_data():
    data = get_monthly_avg_delay()
    return jsonify(data)

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/optimization', methods=['GET', 'POST'])
def optimization():
    result = None
    warning = None
    if request.method == 'POST':
        effort_budget = request.form.get('effort_budget')
        try:
            effort_budget_int = int(effort_budget)
            if effort_budget_int < 1:
                raise ValueError('Effort budget must be a positive integer.')
        except (ValueError, TypeError):
            warning = 'Please enter a valid integer value for effort budget.'
            return render_template('optimization.html', result=None, warning=warning)

        # Call the optimization logic
        result, plateau = get_optimization_result(effort_budget_int)
        if effort_budget_int > plateau:
            warning = f'Effort budget more than the plateau value ({plateau}) will not increase the delay minutes reduced.'
        return render_template('optimization.html', result=result, warning=warning)
    return render_template('optimization.html', result=None, warning=None)

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/next-steps')
def next_steps():
    return render_template('next_steps.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050) 