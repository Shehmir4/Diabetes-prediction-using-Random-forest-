import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import random


# load the dataset
recipes = pd.read_csv(r'C:\Users\mirmu\Django-Projects\diabetespredictionml\diabetespred\dataset\recipes.csv')

# define a function to recommend recipes
def recommend_recipes(meal_type,activity,weight_loss_plan,gender):
    min_calories = 800
    max_calories = 2500  
    min_fat_content = 20
    max_fat_content = 35
    min_carbohydrate_content = 45
    max_carbohydrate_content = 65
    min_protein_content = 10
    max_protein_content = 35
    

    # Generate random values within the recommended ranges
    calories = round(random.uniform(min_calories, max_calories), 2)
    fat_content = round(random.uniform(min_fat_content, max_fat_content), 2)
    carbohydrate_content = round(random.uniform(min_carbohydrate_content, max_carbohydrate_content), 2)
    protein_content = round(random.uniform(min_protein_content, max_protein_content), 2)
    Fiber_Content = round(random.uniform(0,20))
    Sugar_Content = round(random.uniform(0,20))
    Saturated_FatContent = round(random.uniform(0,20))
    Cholesterol_Content =round(random.uniform(10,200))
    Sodium_Content =round(random.uniform(0,20))

    # pre-process the data
    df = recipes.fillna(0) # replace missing values with 0
    recipe_features = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
                    'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 
                    'ProteinContent']                  # select the recipe features
    x = df[recipe_features].values                     # convert the selected features to a NumPy array
    
    
    scaler = MinMaxScaler()                            # initialize a MinMaxScaler
    X_scaled = scaler.fit_transform(x)                 # scale the data


    # train the k-NN model
    k = 5 # number of nearest neighbors to recommend
    model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    model.fit(X_scaled)

    # create a dictionary with user inputs
    user_inputs = {'Calories': [calories], 'FatContent': [fat_content], 
                   'CarbohydrateContent': [carbohydrate_content], 'ProteinContent': [protein_content],'FiberContent':[Fiber_Content],'SugarContent':[Sugar_Content],
                   'SaturatedFatContent':[Saturated_FatContent],'CholesterolContent':[Cholesterol_Content],'SodiumContent':[Sodium_Content]}
    user_inputs_df = pd.DataFrame(user_inputs)                                                                 

    user_inputs_scaled = scaler.transform(user_inputs_df)   
                                                         
    _, indices = model.kneighbors(user_inputs_scaled)                                                            
    recommended_recipes = df.iloc[indices[0]]   
    print(recommended_recipes.columns)                                                                              
    recommended_recipes = recommended_recipes[['Name','Calories']].values.tolist()        
    print(recommended_recipes)           
                                              
   

    return recommended_recipes



