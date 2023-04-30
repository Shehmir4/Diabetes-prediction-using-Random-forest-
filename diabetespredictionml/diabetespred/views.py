import sys
import os
import pandas as pd
import numpy as np
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
from .models import User, metrics ,PredictionForm
from django.shortcuts import render, redirect ,HttpResponse
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .forms import PatientRegistrationForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dietrecomend import dietmlmodel
from django.views.decorators.cache import never_cache
from django.contrib.auth import logout
import traceback

def register_patient(request):
    form = PatientRegistrationForm()
    if request.method == 'POST':
        print(request.POST)
        form = PatientRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Save additional fields for the patient
            user = User(user=user, dob=form.cleaned_data.get('dob'), phone_number=form.cleaned_data.get('phone_number'), address=form.cleaned_data.get('address'))
            user.save()
            return redirect('home')
        else:
            print(form.errors)

    return render(request, 'register_patient.html', {'form': form})


def login_patient(request):
    form = AuthenticationForm()
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, 'Invalid username or password.')

    return render(request, 'login_patient.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('home')

# Create your views here.
@login_required()
def home(request):
    
    return render(request,"base.html")

@login_required()
def modify_details(request):
    user = request.user

    # Retrieve the user's current metrics data from the database
    try:
        metrics_data = metrics.objects.get(user=user)
    except metrics.DoesNotExist:
        # If the user doesn't have any metrics data yet, redirect to the diabetes prediction page
        return redirect('diabetes_prediction')

    if request.method == 'POST':
        height = request.POST.get('Height')
        weight = request.POST.get('Weight')
        age = request.POST.get('Age')
        Insulin = request.POST.get('Insulin')
        SkinThickness = request.POST.get('SkinThickness')
        Glucose = request.POST.get('Glucose')
        BloodPressure = request.POST.get('BloodPressure')
        Pregnancies  = request.POST.get('Pregnancies')

        # Update the metrics data with the new values from the form
        metrics_data.height = height
        metrics_data.weight = weight
        metrics_data.age = age
        metrics_data.BloodPressure = BloodPressure
        metrics_data.Pregnancies = Pregnancies
        metrics_data.SkinThickness = SkinThickness
        metrics_data.Glucose = request.POST.get('Glucose')
        metrics_data.Insulin = Insulin

        metrics_data.save()

        # Redirect back to the modify details page
        return redirect('dataexist')

    return render(request, 'modify_details.html', {'metrics_data': metrics_data})



@login_required()
def diabetes_prediction(request):
    try:
        if request.method == 'POST':
            form = PredictionForm(request.POST)
            if form.is_valid():
                pregnancies = form.cleaned_data['Pregnancies']
                glucose = form.cleaned_data['Glucose']
                blood_pressure = form.cleaned_data['BloodPressure']
                skin_thickness = form.cleaned_data['SkinThickness']
                insulin = form.cleaned_data['Insulin']
                height = form.cleaned_data['Height']
                age = form.cleaned_data['Age']
                weight = form.cleaned_data['Weight']
                user = request.user
                metrics_data = metrics.objects.filter(user=user)

                if metrics_data.exists():
                 
                    return render(request, 'confirm_modify.html')
                
                # Check if any of the input values are NaN (not a number)
                if any(np.isnan([pregnancies, glucose, blood_pressure, skin_thickness, insulin, height, age, weight])):
                    raise ValueError('Invalid input value')

                # Calculate BMI using weight, height, and age
                height_m = height / 100  # convert height from cm to m
                BMI = weight / (height_m ** 2)

                # Save the metrics data to the database
                metrics_data = metrics.objects.create(
                    user=user,
                    age=age,
                    height=height,
                    weight =weight,
                    Pregnancies =  pregnancies,
                    Glucose = glucose,
                    BloodPressure = blood_pressure,
                    SkinThickness = skin_thickness
                )
                metrics_data.save()

                user_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, age, BMI]).reshape(1, -1)
                prediction = predicting(request, user_data)

                return render(request, 'prediction.html', {'prediction': prediction})
            else:
             
                return render(request, 'invalidinput.html', {'error': 'Invalid input'})
       
        else:
            form = PredictionForm()
       
            return render(request, 'base.html', {'form': form})

    except (ValueError, TypeError):
        print("try")
        print(traceback.format_exc())
        return render(request, 'base.html', {'error': 'Invalid input'})
                        
def dataexist(request):
    user = request.user
    metrics_data = metrics.objects.filter(user=user)
    metrics_data = metrics.objects.get(user=user)
    user_data = np.array([metrics_data.Pregnancies,
                            metrics_data.Glucose,
                            metrics_data.BloodPressure,
                            metrics_data.SkinThickness,
                            metrics_data.Insulin,
                            metrics_data.age,
                            metrics_data.height,
                            metrics_data.weight]).reshape(1, -1)
    print(user_data)
    prediction = predicting(request, user_data)
    y_pred = prediction[0]
    return render(request, 'prediction.html', {'prediction': prediction})   

@login_required()
def predicting(request,user_data):
    df = pd.read_csv(r"C:\Users\mirmu\Django-Projects\diabetespredictionml\diabetespred\dataset\diabetes.csv")          # Reading CSV Files 
     # Calculate BMI using weight and height from user data
  
    height_m = user_data[0][6] / 100  # convert height from cm to m
    BMI = user_data[0][7] / (height_m ** 2)

    # Add BMI to user data array
    user_data[0][7] = BMI
    user_data = np.delete(user_data, 6, axis=1)
  
    # Split the data into training and testing sets
    X = df.drop(['Outcome','DiabetesPedigreeFunction'], axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier on the training data
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test data and compute the accuracy
    test_accuracy = clf.score(X_test, y_test)
    print(test_accuracy)
    
    y_pred = clf.predict(user_data)


    return y_pred
   
@login_required()


@login_required
def diet_recomendation(request):
    if request.method == 'POST':
        gender = request.POST.get('gender')
        activity = int(request.POST.get('activity'))
        weight_loss_plan = request.POST.get('weight-loss-plan')
        meal_type = request.POST.get('meal_type')
        
        try:
            # Retrieve the metrics object for the currently logged-in user
            metrics_data = metrics.objects.get(user=request.user)
            age  = metrics_data.age
            height = metrics_data.height
            weight = metrics_data.weight

            dietsuggestions =  dietmlmodel.recommend_recipes(gender,activity,weight_loss_plan,meal_type,age,height,weight)
         
            return render(request, 'dietrecomendation.html',{'dietsuggestion':dietsuggestions[0],'BMI':dietsuggestions[1]})
        except metrics.DoesNotExist:
            # Handle the case when metrics object does not exist for the currently logged-in user
            return HttpResponse("Metrics data does not exist for the current user.")
    else:
        return render(request, 'dietrecomendation.html')


    





