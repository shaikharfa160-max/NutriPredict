import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# FILE PATHS

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "calorie_model.pkl"
)

DIET_DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "diet_recommendation.csv"
)

# LOAD ML MODEL

model_data = None

if os.path.exists(MODEL_PATH):
    try:
        model_data = joblib.load(MODEL_PATH)
        print("ML Model successfully loaded!")
    except Exception as e:
        print(f"Error loading ML Model: {e}")


# LOAD DIET RECOMMENDATION DATASET

diet_df = None

if os.path.exists(DIET_DATASET_PATH):
    try:
        diet_df = pd.read_csv(DIET_DATASET_PATH)

        print("Diet Recommendation Dataset successfully loaded!")
        print(f"Diet Dataset Shape: {diet_df.shape}")

    except Exception as e:
        print(f"Error loading Diet Recommendation Dataset: {e}")

else:
    print("Diet Recommendation Dataset not found!")


# ACTIVITY NAME

def get_activity_name(value):
    """
    Convert activity multiplier from browser
    to activity name used in Dataset 1.
    """

    activity_values = {
        "1.2": "Sedentary",
        "1.375": "Lightly Active",
        "1.55": "Moderately Active",
        "1.725": "Very Active",
        "1.9": "Athlete"
    }

    return activity_values.get(
        str(value),
        "Moderately Active"
    )


# BMI STATUS

def bmi_status(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Healthy"

    elif bmi < 30:
        return "Overweight"

    return "Obese"


# FIND DIET RECOMMENDATION

def get_diet_recommendation(target_calories, selected_diet):

    if diet_df is None or diet_df.empty:
        return None

    try:

        data = diet_df.copy()

        data["Daily_Calories"] = pd.to_numeric(
            data["Daily_Calories"],
            errors="coerce"
        )

        data = data.dropna(
            subset=["Daily_Calories"]
        )

        # Use only rows from the diet selected by the user
        data = data[
            data["Diet_Type"].astype(str).str.strip().str.lower()
            == str(selected_diet).strip().lower()
        ]

        if data.empty:
            print(
                f"No meal records found for diet: {selected_diet}"
            )
            return None

        # Find the meal-plan row whose whole-day calorie value
        # is closest to the model's predicted requirement
        data["calorie_difference"] = (
            data["Daily_Calories"] - target_calories
        ).abs()

        recommended_row = data.sort_values(
            "calorie_difference"
        ).iloc[0]

        return {
            "breakfast": str(recommended_row["Breakfast"]),
            "lunch": str(recommended_row["Lunch"]),
            "dinner": str(recommended_row["Dinner"]),
            "snacks": str(recommended_row["Snacks"]),
            "water": float(recommended_row["Water_Intake_L"]),
            "dataset_calories": int(recommended_row["Daily_Calories"]),
            "diet_type": str(recommended_row["Diet_Type"])
        }

    except Exception as e:

        print(
            f"Error generating diet recommendation: {e}"
        )

        return None


# HOME

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# PREDICTION

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(force=True)

        # USER INPUT
        
        age = float(
            data.get("age")
        )

        gender_name = data.get(
            "gender",
            "Female"
        )

        height = float(
            data.get("height")
        )

        weight = float(
            data.get("weight")
        )

        activity_value = data.get(
            "activity",
            "1.55"
        )

        diet = data.get(
            "diet",
            "Balanced"
        )

        # VALIDATION
      
        if age <= 0:
            raise ValueError(
                "Please enter a valid age."
            )

        if height <= 0:
            raise ValueError(
                "Please enter a valid height."
            )

        if weight <= 0:
            raise ValueError(
                "Please enter a valid weight."
            )

        # BMI
       
        bmi = weight / (
            (height / 100) ** 2
        )

        activity_name = get_activity_name(
            activity_value
        )

        # RANDOM FOREST MODEL
      
        if model_data is None:

            raise ValueError(
                "ML model could not be loaded."
            )

        model = model_data["model"]

        gender_map = model_data[
            "gender_map"
        ]

        activity_map = model_data[
            "activity_map"
        ]

        gender_encoded = gender_map.get(
            gender_name,
            0
        )

        activity_encoded = activity_map.get(
            activity_name,
            2
        )

        # SIX FEATURES
        
        features = np.array([[

            age,

            gender_encoded,

            height,

            weight,

            bmi,

            activity_encoded

        ]])

        predicted_calories = float(
            model.predict(features)[0]
        )

        # FINAL CALORIE REQUIREMENT
        
        # Use the Random Forest prediction directly.
        # No manual +/-400 calorie adjustment is applied.

        target_calories = round(predicted_calories)

        # MACRONUTRIENTS
    
        protein_g = round(
            (target_calories * 0.25) / 4
        )

        carbs_g = round(
            (target_calories * 0.50) / 4
        )

        fats_g = round(
            (target_calories * 0.25) / 9
        )

        # DIET DATASET RECOMMENDATION
      
        recommendation = get_diet_recommendation(
            target_calories,
            diet
        )

        # MEAL PLAN
        
        meals = []

        water_liters = round(
            weight * 0.043,
            1
        )

        if recommendation:

            meals = [
                {
                    "title": "🌅 Breakfast",
                    "desc": recommendation["breakfast"]
                },
                {
                    "title": "🥗 Lunch",
                    "desc": recommendation["lunch"]
                },
                {
                    "title": "🍏 Snacks",
                    "desc": recommendation["snacks"]
                },
                {
                    "title": "🍲 Dinner",
                    "desc": recommendation["dinner"]
                }
            ]

            water_liters = recommendation["water"]

        # RESPONSE
        
        return jsonify({

            "status": "success",

            "calories": target_calories,

            "bmi": round(
                bmi,
                1
            ),

            "health_status": bmi_status(
                bmi
            ),

            "water": water_liters,

            "protein": protein_g,

            "carbs": carbs_g,

            "fats": fats_g,

            "protein_percent": 25,

            "carbs_percent": 50,

            "fats_percent": 25,

            "diet": diet,

            "model": "Random Forest Regression",

            "meals": meals

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 400


# RUN APPLICATION

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )