# 🚗 Car Price Prediction with Machine Learning

URL: [Webpage](https://car-price-prediction-ui.onrender.com)

Welcome to the Car Price Prediction project! This repository provides a complete pipeline for predicting used car prices using machine learning, including data cleaning, visualization, model training, API deployment, and a user-friendly Streamlit web app.

---

## 📂 Project Structure

```
CAR-PRICE-PREDICTION-WITH-MACHINE-LEARNING/
│
├── Api/                # FastAPI backend for model inference
│   └── main.py
│
├──Models/
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── Streamlit_app/      # Streamlit frontend for user interaction
│   └── app.py
│
├── Data/               # Datasets
│   ├── car_data.csv
│   └── cleaned_car_data.csv
│
├── Notebooks/          # Jupyter notebooks for EDA, cleaning, training
│   ├── data_clean.ipynb
│   ├── data_visualization.ipynb
│   └── model_training.ipynb
│   
├── requirements.txt
├── render.yaml         # (Optional) Render deployment config
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CAR-PRICE-PREDICTION-WITH-MACHINE-LEARNING.git
cd CAR-PRICE-PREDICTION-WITH-MACHINE-LEARNING
```

### 2. Install Dependencies

Install Python 3.12+ and pip, then:

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Backend

```bash
cd Api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the Streamlit Frontend

Open a new terminal:

```bash
cd Streamlit_app
streamlit run app.py
```

### 5. Use the App

- Go to [http://localhost:8501](http://localhost:8501) in your browser.
- Enter car details and get instant price predictions from multiple models.
- See feature importances and compare model performances.

---

## 🧑‍💻 Features

- **Data Cleaning:** Remove outliers, filter non-car entries, handle missing values.
- **Visualization:** Explore data distributions and relationships with Seaborn and Matplotlib.
- **Model Training:** Train and compare Linear Regression, Random Forest, and XGBoost models.
- **API:** FastAPI backend serves predictions and model confidences.
- **Frontend:** Streamlit app with car name suggestions, input validation, and attractive UI.
- **Model Selection:** Automatically selects the best model based on confidence (R² score).
- **Deployment Ready:** Easily deploy both backend and frontend on [Render](https://render.com) or similar platforms.

---

## 📊 Data Columns

- `Car_Name`: Name of the car (with suggestions in the UI)
- `Year`: Year of purchase
- `Selling_Price`: Price at which the car is being sold (target)
- `Present_Price`: Current ex-showroom price
- `Driven_kms`: Kilometers driven
- `Fuel_Type`: Petrol, Diesel, CNG, etc.
- `Selling_type`: Dealer or Individual
- `Transmission`: Manual or Automatic
- `Owner`: Number of previous owners

---

## 🏆 Model Performance

| Model              | MAE  | MSE  | R² Score (%) |
|--------------------|------|------|--------------|
| Linear Regression  | 1.36 | 3.1  | 92.44        |
| Random Forest      | 1.19 | 3.1  | 92.43        |
| XGBoost            | 1.58 | 5.6  | 86.35        |

*Random Forest is selected as the best model by default.*

---

## 🌐 Deployment on Render

1. **Push your code to GitHub.**
2. **Create two Render web services:**
   - One for the FastAPI backend (`Api/`)
   - One for the Streamlit frontend (`Streamlit_app/`)
3. **Set the correct start commands:**
   - FastAPI: `uvicorn main:app --host 0.0.0.0 --port 10000`
   - Streamlit: `streamlit run app.py --server.port 10001 --server.address 0.0.0.0`
4. **Set environment variables if needed (e.g., API URL for frontend).**
5. **See `render.yaml` for an example configuration.**

---

## 📒 Notebooks

- **data_clean.ipynb:** Data cleaning, outlier detection (IQR method), and preprocessing.
- **data_visualization.ipynb:** Distribution plots, scatter plots, and correlation heatmaps.
- **model_training.ipynb:** Model training, evaluation, and feature importance plots.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📢 Acknowledgements

- [Streamlit](https://streamlit.io/) for the frontend.
- [FastAPI](https://fastapi.tiangolo.com/) for the backend.
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/) for machine learning.
- [Render](https://render.com/) for easy deployment.

---

## 📬 Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

*Happy Predicting! 🚗*