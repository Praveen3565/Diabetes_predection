# Diabetes Prediction Using Hybrid Models

## Project Overview
This project aims to develop a predictive model for diabetes diagnosis using a hybrid approach that combines XGBoost and Deep Neural Networks (DNN). The model is trained on a dataset containing various health parameters related to diabetes. The project employs Particle Swarm Optimization (PSO) to optimize the hyperparameters of the XGBoost model.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- XGBoost
- TensorFlow/Keras
- PySwarm

## Dataset
The dataset used for this project is a CSV file containing 2000 rows and 9 columns related to diabetes, including:
- **Pregnancies**: Number of pregnancies
- **Glucose**: Glucose level
- **BloodPressure**: Blood pressure value
- **SkinThickness**: Thickness of the skin
- **Insulin**: Insulin level
- **BMI**: Body Mass Index
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history
- **Age**: Age of the patient
- **Outcome**: Target variable indicating whether the patient has diabetes (1) or not (0)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd diabetes-prediction
   ```

2. Install the required packages:
   ```bash
   pip install pandas scikit-learn xgboost tensorflow pyswarm
   ```

## Usage
1. Ensure you have the dataset (`diabetes_new_update.csv`) in the specified directory.
2. Run the Python script to train the model and evaluate its performance:
   ```bash
   python diabetes_prediction.py
   ```

## Model Training
- The project uses Particle Swarm Optimization (PSO) to optimize the hyperparameters for the XGBoost model.
- A Deep Neural Network (DNN) is also trained for comparison.
- Predictions from both models are combined to create a hybrid model.

## Model Evaluation
The model's performance is evaluated based on accuracy metrics, and the results are printed to the console. 

## Model Saving
Trained models are saved in the following formats:
- XGBoost model: `xgb_diabetes_model.json`
- DNN model: `dnn_diabetes_model.h5`

## Contributing
Contributions are welcome! If you have suggestions or improvements, please feel free to submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify any sections as needed!
