import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import MinMaxScaler ,LabelEncoder


# load the dataset
recipes = pd.read_csv(r'C:\Users\mirmu\Django-Projects\diabetespredictionml\diabetespred\dataset\input.csv')


# define a function to recommend recipes
def recommend_recipes(gender, activity, weight_loss_plan, meal_type, age, height, weight):
    height_m = height / 100  # convert height from cm to m
    BMI = weight / (height_m ** 2)

    # check for invalid inputs
    if meal_type not in ["breakfast", "lunch", "dinner"]:
        print("Invalid meal type input")
        return

    nutrient_ranges = {
        18.5: get_nutrient_ranges(18.5),
        25.0: get_nutrient_ranges(25),
        30.0: get_nutrient_ranges(30),
        31.0: get_nutrient_ranges(31)
    }

    if BMI < 18.5:
        message = "Your BMI is {:.1f}, which falls within the underweight range.".format(BMI)
        inputvalues = list(nutrient_ranges[18.5])
    elif BMI < 25.0:
        message = "Your BMI is {:.1f}, which falls within the healthyweight range.".format(BMI)
        inputvalues = list(nutrient_ranges[25.0])
    elif BMI < 30.0:
          
        message = "Your BMI is {:.1f}, which falls within the overweight range.".format(BMI)

        inputvalues = list(nutrient_ranges[30.0])
    else:
        message = "Your BMI is {:.1f}, which falls within the obesity range.".format(BMI)
        inputvalues = list(nutrient_ranges[31.0])
    meal_data = recipes[recipes[meal_type.capitalize()] == 1]  # use meal_type to filter meal_data
    recommendations = knn_recommendations(meal_data,inputvalues)
    return recommendations,message

def knn_recommendations(meal_data, input_values):
    le = LabelEncoder()
    scaler = MinMaxScaler()
    y = le.fit_transform(meal_data['Food_items'])# target column
    x = meal_data.drop(['Food_items','Lunch','Breakfast','Dinner','VegNovVeg'], axis=1)  # drop the 'food_items' column
    scaled_df = scaler.fit_transform(x)  # fit and transform the dataset using the scaler
    input_array = np.array(input_values).reshape(1, -1)
    standardized_input = scaler.transform(input_array)  # transform the input values using the scaler  
    model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    model.fit(scaled_df)
    # Fit a KNN model to the training data

    _, indices = model.kneighbors(standardized_input)  
 
    nearest_neighbors = meal_data.iloc[indices[0]]
    recommended_food_items = nearest_neighbors['Food_items'].tolist()

    return recommended_food_items

def get_nutrient_ranges(bmi):
    if bmi <= 18.5:
        min_calories, max_calories = 1900, 2500
        min_fat_content, max_fat_content = 20, 35
        min_carbohydrate_content, max_carbohydrate_content = 250, 300
        min_protein_content, max_protein_content = 10, 35
        min_sugars_content, max_sugars_content = 25, 38
    elif bmi <= 25:
        min_calories, max_calories = 1500, 2200
        min_fat_content, max_fat_content = 15, 30
        min_carbohydrate_content, max_carbohydrate_content = 200, 300
        min_protein_content, max_protein_content = 15, 30
        min_sugars_content, max_sugars_content = 25, 40
    elif bmi <=30:
        min_calories, max_calories = 1000, 1500
        min_fat_content, max_fat_content = 10, 28
        min_carbohydrate_content, max_carbohydrate_content = 200, 250
        min_protein_content, max_protein_content = 20, 30
        min_sugars_content, max_sugars_content = 5, 10
    else:
        min_calories, max_calories = 1000, 1300
        min_fat_content, max_fat_content = 10, 15
        min_carbohydrate_content, max_carbohydrate_content = 200, 250
        min_protein_content, max_protein_content = 20, 30
        min_sugars_content, max_sugars_content = 5, 10

    # Generate random values within the recommended ranges
    calories = round(random.uniform(min_calories, max_calories), 2)
    fat_content = round(random.uniform(min_fat_content, max_fat_content), 2)
    carbohydrates = round(random.uniform(min_carbohydrate_content, max_carbohydrate_content), 2)   #Grams 
    proteins = round(random.uniform(min_protein_content, max_protein_content), 2)
    sugars = round(random.uniform(min_sugars_content, max_sugars_content))
    fibre = round(random.uniform(25,38))
    iron = round(random.uniform(15,18))
    calcium = round(random.uniform(800,1000))
    sodium = round(random.uniform(1500,2300))
    potassium = round(random.uniform(2000,3000)) 
    vitamin = round(random.uniform(600,800))

    return calories, fat_content, carbohydrates, proteins, sugars, fibre, iron, calcium, sodium, potassium, vitamin




















