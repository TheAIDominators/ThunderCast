from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import random
import numpy as np
from datetime import datetime

# Add the Model directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))

predictor = None
model_loaded = False
windspeed_model_data = None
windspeed_model_loaded = False

try:
    from thunderstorm_prediction_model import ThunderstormPredictor
    import joblib

    # Load thunderstorm model
    predictor = ThunderstormPredictor()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'Model', 'thunderstorm_model.joblib')
    if os.path.exists(model_path):
        predictor.load_model(model_path)
        print("✅ Thunderstorm model loaded successfully!")
        model_loaded = True
    else:
        print("⚠️ Thunderstorm model file not found")
        model_loaded = False

    # Load windspeed model
    windspeed_model_path = os.path.join(os.path.dirname(__file__), '..', 'Model', 'windspeed_model.joblib')
    if os.path.exists(windspeed_model_path):
        windspeed_model_data = joblib.load(windspeed_model_path)
        print("✅ Windspeed model loaded successfully!")
        windspeed_model_loaded = True
    else:
        print("⚠️ Windspeed model file not found")
        windspeed_model_loaded = False
except Exception as e:
    print(f"❌ Error during model loading: {e}")

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "message": "Hackovate Weather Prediction API",
        "models": {
            "thunderstorm": model_loaded,
            "windspeed": windspeed_model_loaded
        },
        "endpoints": {
            "thunderstorm_predict": "/api/ml/predict",
            "windspeed_predict": "/api/windspeed/predict",
            "health": "/api/health"
        }
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "OK",
        "thunderstorm_model": model_loaded,
        "windspeed_model": windspeed_model_loaded,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/ml/predict', methods=['POST'])
def predict_thunderstorm():
    try:
        data = request.get_json()

        required_params = [
            'wind_sfc_speed_ms', 'wind_sfc_dir_deg', 'wind_500_speed_ms', 'wind_500_dir_deg',
            'temp_2m_C', 'temp_500_C', 'rh_2m_pct', 'pressure_sfc_hPa',
            'precipitable_water_mm', 'cloud_cover_frac', 'cloud_top_temp_C',
            'CAPE_Jkg', 'Lifted_Index_C', 'K_index', 'shear_850_500_ms'
        ]

        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return jsonify({
                "success": False,
                "error": f"Missing parameters: {missing_params}"
            }), 400

        if predictor and model_loaded:
            result = predictor.predict_thunderstorm(data)
        else:
            result = fallback_thunderstorm_prediction(data)

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/windspeed/predict', methods=['POST'])
def predict_windspeed():
    try:
        data = request.get_json()

        # Your windspeed parameters
        required_params = [
            'IND', 'RAIN', 'IND.1', 'T.MAX', 'IND.2', 'T.MIN.G',
            'wind_lag_1', 'wind_lag_2', 'wind_lag_3',
            'ma_3', 'ma_5', 'ma_7',
            'std_3', 'std_5', 'std_7'
        ]

        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return jsonify({
                "success": False,
                "error": f"Missing windspeed parameters: {missing_params}"
            }), 400

        if windspeed_model_loaded:
            model = windspeed_model_data['model']
            scaler = windspeed_model_data['scaler']
            feature_names = windspeed_model_data['feature_names']

            input_array = np.array([[data[feature] for feature in feature_names]])
            input_scaled = scaler.transform(input_array)
            predicted_windspeed = model.predict(input_scaled)[0]

            if predicted_windspeed < 5:
                wind_category = "Light"
                alert = f"Light winds: {predicted_windspeed:.1f} m/s - Calm conditions"
            elif predicted_windspeed < 10:
                wind_category = "Moderate"
                alert = f"Moderate winds: {predicted_windspeed:.1f} m/s - Normal conditions"
            elif predicted_windspeed < 15:
                wind_category = "Strong"
                alert = f"Strong winds: {predicted_windspeed:.1f} m/s - Be cautious"
            else:
                wind_category = "Very Strong"
                alert = f"Very strong winds: {predicted_windspeed:.1f} m/s - High wind warning"

            result = {
                'predicted_windspeed': float(predicted_windspeed),
                'wind_category': wind_category,
                'alert': alert,
                'modelType': 'Random Forest Regressor (Trained)'
            }
        else:
            result = fallback_windspeed_prediction(data)

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/ml/predict/<location_id>')
def predict_thunderstorm_for_location(location_id):
    try:
        location_data = {
            'wind_sfc_speed_ms': 12.5, 'wind_sfc_dir_deg': 225, 'wind_500_speed_ms': 18.2, 'wind_500_dir_deg': 230,
            'temp_2m_C': 35.5, 'temp_500_C': -8.2, 'rh_2m_pct': 85, 'pressure_sfc_hPa': 995,
            'precipitable_water_mm': 45, 'cloud_cover_frac': 0.8, 'cloud_top_temp_C': -45,
            'CAPE_Jkg': 3200, 'Lifted_Index_C': -6.5, 'K_index': 35, 'shear_850_500_ms': 18
        }

        if predictor and model_loaded:
            result = predictor.predict_thunderstorm(location_data)
        else:
            result = fallback_thunderstorm_prediction(location_data)

        return jsonify({
            "success": True,
            "data": {
                **result,
                "locationId": location_id
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "locationId": location_id
        }), 500

@app.route('/api/windspeed/predict/<location_id>')
def predict_windspeed_for_location(location_id):
    try:
        location_windspeed_data = {
            'IND': 1.2, 'RAIN': 0.5, 'IND.1': 0.8, 'T.MAX': 25.0, 'IND.2': 1.1, 'T.MIN.G': 15.0,
            'wind_lag_1': 8.5, 'wind_lag_2': 7.2, 'wind_lag_3': 9.1,
            'ma_3': 8.3, 'ma_5': 8.1, 'ma_7': 7.9,
            'std_3': 1.2, 'std_5': 1.5, 'std_7': 1.8
        }

        if windspeed_model_loaded:
            model = windspeed_model_data['model']
            scaler = windspeed_model_data['scaler']
            feature_names = windspeed_model_data['feature_names']

            input_array = np.array([[location_windspeed_data[feature] for feature in feature_names]])
            input_scaled = scaler.transform(input_array)
            predicted_windspeed = model.predict(input_scaled)[0]

            if predicted_windspeed < 5:
                wind_category = "Light"
            elif predicted_windspeed < 10:
                wind_category = "Moderate"
            elif predicted_windspeed < 15:
                wind_category = "Strong"
            else:
                wind_category = "Very Strong"

            result = {
                'predicted_windspeed': float(predicted_windspeed),
                'wind_category': wind_category,
                'alert': f"{wind_category} winds: {predicted_windspeed:.1f} m/s",
                'locationId': location_id
            }
        else:
            result = fallback_windspeed_prediction(location_windspeed_data)
            result['locationId'] = location_id

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "locationId": location_id
        }), 500

def fallback_thunderstorm_prediction(parameters):
    """Fallback thunderstorm prediction"""
    cape = parameters.get('CAPE_Jkg', 1000)
    lifted_index = parameters.get('Lifted_Index_C', 0)
    wind_shear = parameters.get('shear_850_500_ms', 10)
    humidity = parameters.get('rh_2m_pct', 70)
    pressure = parameters.get('pressure_sfc_hPa', 1000)

    risk_score = 0.1
    if cape > 2500:
        risk_score += 0.3
    elif cape > 1000:
        risk_score += 0.2

    if lifted_index < -4:
        risk_score += 0.2
    elif lifted_index < -2:
        risk_score += 0.1

    if humidity > 85:
        risk_score += 0.1
    if pressure < 995:
        risk_score += 0.1

    probability = min(0.95, max(0.05, risk_score + random.uniform(-0.05, 0.05)))

    if probability > 0.75:
        risk_level = "Red"
        alert = f"SEVERE: {probability*100:.1f}% thunderstorm probability - Dangerous conditions likely"
    elif probability > 0.50:
        risk_level = "Yellow"
        alert = f"MODERATE: {probability*100:.1f}% thunderstorm probability - Monitor conditions"
    elif probability > 0.25:
        risk_level = "Yellow"
        alert = f"LOW-MODERATE: {probability*100:.1f}% thunderstorm risk"
    else:
        risk_level = "Green"
        alert = f"LOW: {probability*100:.1f}% thunderstorm risk - Conditions favorable"

    return {
        "prediction": 1 if probability > 0.5 else 0,
        "probability": round(probability, 3),
        "confidence": random.randint(75, 95),
        "risk_level": risk_level,
        "riskLevel": risk_level,
        "alert": alert,
        "modelType": "Enhanced Fallback Prediction Model",
        "factors": {
            "cape": cape,
            "lifted_index": lifted_index,
            "wind_shear": wind_shear,
            "humidity": humidity,
            "pressure": pressure
        }
    }

def fallback_windspeed_prediction(parameters):
    """Fallback windspeed prediction when model is not available"""
    base_windspeed = 8.0
    temp_factor = parameters.get('T.MAX', 20) / 20
    wind_lag_avg = (parameters.get('wind_lag_1', 8) + parameters.get('wind_lag_2', 8) + parameters.get('wind_lag_3', 8)) / 3

    predicted_windspeed = base_windspeed * temp_factor * 0.3 + wind_lag_avg * 0.7
    predicted_windspeed = max(0, min(25, predicted_windspeed))

    if predicted_windspeed < 5:
        wind_category = "Light"
    elif predicted_windspeed < 10:
        wind_category = "Moderate"
    elif predicted_windspeed < 15:
        wind_category = "Strong"
    else:
        wind_category = "Very Strong"

    return {
        'predicted_windspeed': float(predicted_windspeed),
        'wind_category': wind_category,
        'alert': f"{wind_category} winds: {predicted_windspeed:.1f} m/s",
        'modelType': 'Fallback Windspeed Model'
    }

if __name__ == '__main__':
    print("🚀 Starting Hackovate Weather Prediction API...")
    if not model_loaded:
        print("⚠️ Thunderstorm model not loaded, using fallback")
    if not windspeed_model_loaded:
        print("⚠️ Windspeed model not loaded, using fallback")
    print("🌐 Server starting on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
