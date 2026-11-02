/* BMI */

function calculateBMI(height, weight) {

    const heightMeters = height / 100;

    return (
        weight / (heightMeters * heightMeters)
    ).toFixed(1);
}


function getBMIStatus(bmi) {

    if (bmi < 18.5) {
        return "Underweight";
    }

    if (bmi < 25) {
        return "Healthy";
    }

    if (bmi < 30) {
        return "Overweight";
    }

    return "Obese";
}


/* GENERATE */

async function generateRecommendation() {

    const age =
        document.getElementById("age").value;

    const gender =
        document.getElementById("gender").value;

    const height =
        document.getElementById("height").value;

    const weight =
        document.getElementById("weight").value;

    const activity =
        document.getElementById("activity").value;

    const diet =
        document.getElementById("diet").value;


    /* VALIDATION */

    if (
        !age ||
        !gender ||
        !height ||
        !weight ||
        !activity ||
        !diet
    ) {

        showError(
            "Please fill in all fields before generating your recommendation."
        );

        return;
    }


    const loading =
        document.getElementById("loading");

    const error =
        document.getElementById("error");

    const results =
        document.getElementById("results");


    loading.style.display = "block";
    error.style.display = "none";
    results.style.display = "none";


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    age: Number(age),
                    gender: gender,
                    height: Number(height),
                    weight: Number(weight),
                    activity: activity,
                    diet: diet
                })
            }
        );


        const data = await response.json();


        if (
            !response.ok ||
            data.status !== "success"
        ) {
            throw new Error(
                data.message ||
                "Unable to generate recommendation."
            );
        }


        /* CALORIES */

        document.getElementById(
            "calories"
        ).textContent = Math.round(data.calories);


        document.getElementById(
            "calorieSub"
        ).textContent =
            "Predicted using Random Forest Regression";


        /* MACROS */

        document.getElementById(
            "protein"
        ).textContent = data.protein;

        document.getElementById(
            "carbs"
        ).textContent = data.carbs;

        document.getElementById(
            "fats"
        ).textContent = data.fats;


        /* BMI */

        document.getElementById(
            "bmi"
        ).textContent = Number(data.bmi).toFixed(1);

        document.getElementById(
            "bmiStatus"
        ).textContent = data.health_status;


        /* WATER */

        document.getElementById(
            "water"
        ).textContent =
            Number(data.water).toFixed(1) + " L";


        /* DIET */

        document.getElementById(
            "dietResult"
        ).textContent = data.diet;

        document.getElementById(
            "dietBadge"
        ).textContent = data.diet;


        /* MEALS */

        const mealList =
            document.getElementById("mealList");

        mealList.innerHTML = "";


        if (
            Array.isArray(data.meals) &&
            data.meals.length > 0
        ) {

            data.meals.forEach(function (meal) {

                const mealDiv =
                    document.createElement("div");

                mealDiv.className = "meal";

                let icon = "🍽️";

                if (meal.title.includes("Breakfast")) {
                    icon = "🌅";
                } else if (meal.title.includes("Lunch")) {
                    icon = "🥗";
                } else if (meal.title.includes("Snack")) {
                    icon = "🍏";
                } else if (meal.title.includes("Dinner")) {
                    icon = "🍲";
                }

                mealDiv.innerHTML = `
                            <div class="meal-icon">
                                ${icon}
                            </div>

                            <div class="meal-content">
                                <h4>${meal.title}</h4>
                                <p>${meal.desc}</p>
                            </div>
                        `;

                mealList.appendChild(mealDiv);

            });

        } else {

            mealList.innerHTML = `
                        <div class="meal">
                            <div class="meal-icon">🍽️</div>
                            <div class="meal-content">
                                <h4>Meal Plan</h4>
                                <p>No meal recommendation was found for the selected diet.</p>
                            </div>
                        </div>
                    `;

        }


        loading.style.display = "none";
        results.style.display = "block";


        setTimeout(function () {

            results.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }, 100);

    }

    catch (err) {

        loading.style.display = "none";

        showError(
            err.message ||
            "Something went wrong. Please try again."
        );

    }

}


/* ERROR */

function showError(message) {

    const error =
        document.getElementById(
            "error"
        );


    error.textContent =
        message;


    error.style.display =
        "block";


    error.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}