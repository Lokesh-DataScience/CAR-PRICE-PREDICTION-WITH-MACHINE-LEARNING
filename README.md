# OASIS_CAR-PRICE-PREDICTION-WITH-MACHINE-LEARNING

**About Oasis :** OASIS INFOBYTE is a community of diverse people coming together with similar objectives and ultimate goals. 
OASIS INFOBYTE is all about creating opportunities for leadership development, learning, student engagement, and fostering of shared interests. We develop enriching environments and experiences that promote students' knowledge and wellbeing.

## Project Overview

The car price model will predict the car price with some given features (e.g. car name, fuel type, owners, etc). 

## Repository Contents

- **Car_price_Pred_model.ipynb**: Jupyter notebook containing the complete code for training and evaluating the Car Price Prediction model.
- **Car_Price_Pred_Web_App.py**: Python script that provides a graphical user interface (GUI) using Streamlit to demonstrate the caar price prediction model.
- **rf_model.pkl**: Pretraained model from `Car_price_Pred_model.ipynb`.
- **requirements.txt**: List of Python libraries required to run `Car_price_Pred_model.ipynb` and `Car_Price_Pred_Web_App.py`.
- **car data.csv**: Dataset of Car-Price-Preidction-Model model.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.12
- pip

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Lokesh-DataScience/oibsip_3.git
    cd oibsip_3
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Demo

To run the GUI demo of the Iris Flower Classification model:

```bash
python Car_Price_Pred_Web_App.py
```
### Usage
- **The Car_Price_Pred_Web_App.py scripts open a window displaying the web page.**
- **The model will take inputs from user and predicts price of car.**
- **The Car_price_Pred_model.ipynb notebook can be used to understand and reproduce the training process of the model.**

### Model Details
- **The car price prediction model  is saved in the rf_model.pkl file.**
- **This model is trained on a dataset of car data capable of predicting car prices.**

### Contributing
- **Contributions are welcome! Please feel free to submit a Pull Request.**

### Acknowledgements
- **Streamlit for providing the tools for creating a web app.**
- **scikit-learn for the machine learning framework.**

