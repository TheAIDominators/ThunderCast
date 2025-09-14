# Thundercast – AI Weather Intelligence

Thundercast is an AI-powered weather platform delivering real-time thunderstorm risk and windspeed forecasts. It combines:
- Python/Flask backend for REST APIs
- Trained ML models (thunderstorm classification + windspeed regression)
- React frontend with a modern Layer Panel, SweetAlert-style modals, and branding

## What we built

- Trained and saved thunderstorm model (RandomForestClassifier) using curated features from CSV; exported `thunderstorm_model.joblib`.
- Trained and saved windspeed model (RandomForestRegressor) using lag/MA/STD features; exported `windspeed_model.joblib`.
- Flask API (port 5001) loading both models, with robust validation and fallbacks.
- Frontend Layer Panel with parameter controls, presets, SweetAlert-style UI, and branding (logo + “Thundercast”).
- Fixed macOS port conflict, improved API error handling, added health endpoint, and end-to-end integration.

## Repository layout

```
Hackovate/
├─ backend/
│  └─ app.py                      # Flask API (thunderstorm + windspeed)
├─ Model/
│  ├─ thunderprediction_model.py  # CSV-based thunderstorm model + plots + joblib
│  ├─ thunderstorm_prediction_model.py # synthetic/fast training (optional)
│  ├─ windspeed_prediction_model.py    # windspeed model + joblib
│  ├─ thunderstorm_model.joblib   # generated (after training)
│  └─ windspeed_model.joblib      # generated (after training)
├─ frontend/
│  ├─ public/
│  │  └─ logo/logo.jpeg           # app logo (used in UI)
│  └─ src/
│     ├─ components/LayerPanel.jsx # UI panel, SweetAlert-style modal, branding
│     └─ services/                # API clients (ml + windspeed)
└─ README.md                      # this file
```

## Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- pip: `flask`, `flask-cors`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`, `seaborn`

Install Python deps:
```
python3 -m pip install flask flask-cors scikit-learn pandas numpy joblib matplotlib seaborn
```

## Train models

From the Model directory:
```
cd Model
# Thunderstorm (CSV-based). Requires thunderstorm_sample_dataset.csv (index_col=time_utc).
python3 thunderprediction_model.py

# Windspeed
python3 windspeed_prediction_model.py

# Optional: synthetic/fast thunderstorm model
python3 thunderstorm_prediction_model.py
```

Outputs:
- thunderstorm_model.joblib, model_info.json, plots (ROC, confusion matrix, hist, feature importance)
- windspeed_model.joblib, windspeed_model_info.json

## Run backend (Flask)

```
cd backend
python3 app.py
```
Notes:
- Runs on http://localhost:5001 (we avoid macOS AirPlay on 5000).
- Health: GET /api/health

## Run frontend (React)

```
cd frontend
npm install
npm start
```
- UI at http://localhost:3000
- Calls API at http://localhost:5001

## API reference

Health
- GET `/api/health`
- Response:
```
{
  "status": "OK",
  "thunderstorm_model": true,
  "windspeed_model": true,
  "timestamp": "2025-01-01T00:00:00Z"
}
```

Thunderstorm Prediction
- POST `/api/ml/predict`
- Body:
```
{
  "wind_sfc_speed_ms": 10, "wind_sfc_dir_deg": 180,
  "wind_500_speed_ms": 15, "wind_500_dir_deg": 180,
  "temp_2m_C": 20, "temp_500_C": -5,
  "rh_2m_pct": 60, "pressure_sfc_hPa": 1013,
  "precipitable_water_mm": 25, "cloud_cover_frac": 0.5, "cloud_top_temp_C": -20,
  "CAPE_Jkg": 1000, "Lifted_Index_C": 0, "K_index": 25, "shear_850_500_ms": 10
}
```
- Response includes: `prediction`, `probability`, `confidence`, `risk_level` (Green/Yellow/Red), `alert`.

Windspeed Prediction
- POST `/api/windspeed/predict`
- Body:
```
{
  "IND": 1.2, "RAIN": 0.5, "IND.1": 0.8, "T.MAX": 25.0, "IND.2": 1.1, "T.MIN.G": 15.0,
  "wind_lag_1": 8.5, "wind_lag_2": 7.2, "wind_lag_3": 9.1,
  "ma_3": 8.3, "ma_5": 8.1, "ma_7": 7.9,
  "std_3": 1.2, "std_5": 1.5, "std_7": 1.8
}
```
- Response includes: `predicted_windspeed`, `wind_category` (Light/Moderate/Strong/Very Strong), `alert`.

cURL examples:
```
curl -X POST http://localhost:5001/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"wind_sfc_speed_ms":10,"wind_sfc_dir_deg":180,"wind_500_speed_ms":15,"wind_500_dir_deg":180,"temp_2m_C":20,"temp_500_C":-5,"rh_2m_pct":60,"pressure_sfc_hPa":1013,"precipitable_water_mm":25,"cloud_cover_frac":0.5,"cloud_top_temp_C":-20,"CAPE_Jkg":1000,"Lifted_Index_C":0,"K_index":25,"shear_850_500_ms":10}'

curl -X POST http://localhost:5001/api/windspeed/predict \
  -H "Content-Type: application/json" \
  -d '{"IND":1.2,"RAIN":0.5,"IND.1":0.8,"T.MAX":25,"IND.2":1.1,"T.MIN.G":15,"wind_lag_1":8.5,"wind_lag_2":7.2,"wind_lag_3":9.1,"ma_3":8.3,"ma_5":8.1,"ma_7":7.9,"std_3":1.2,"std_5":1.5,"std_7":1.8}'
```

## Frontend UX

- LayerPanel.jsx: parameter groups, presets, and a Run Prediction button.
- SweetAlert-style modal shows success/warning/error styled to risk level.
- Branding: logo at `frontend/public/logo/logo.jpeg` + “Thundercast” title; document title set.

## Troubleshooting

- Port already in use on 5000: backend uses 5001; ensure nothing occupies 5001.
- CSV not found for thunderstorm training: place `thunderstorm_sample_dataset.csv` in `Model/` with `index_col=time_utc`.
- CORS: Flask-CORS enabled; frontend uses http://localhost:5001.
- Matplotlib display: scripts save PNGs to disk; use non-interactive backend if needed.

## Notes

- Models are saved with scaler and feature order for consistent inference.
- Backend includes graceful fallbacks if a model file is missing.
- The UI is designed for easy extension (new layers/models).

## License

Proprietary. All rights reserved.
Contact
For issues and enhancements, edit backend/app.py, Model scripts, or LayerPanel.jsx and restart the services.
