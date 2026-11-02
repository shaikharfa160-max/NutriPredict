# NutriPredict
A web app that predicts a person's daily calorie requirement using Machine Learning and suggests a matching diet plan.

## Features
- Predicts daily calories using a Random Forest model
- Compares Random Forest with Multiple Linear Regression (MAE, RMSE, R2)
- Shows BMI status, protein, carbs, fats and water intake
- Suggests a meal plan (breakfast, lunch, snacks, dinner) for the selected diet type

## Tech Stack
Python, Flask, Scikit-learn, Pandas, NumPy, Matplotlib, HTML, CSS, JavaScript

## How to run
pip install -r requirements.txt
python app.py

Then open http://127.0.0.1:5000 in your browser.

To train the model again:
python train_model.py