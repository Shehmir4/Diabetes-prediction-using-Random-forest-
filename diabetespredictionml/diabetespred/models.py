from django.db import models
from django.contrib.auth.models import User
from django import forms


class metrics(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    age = models.FloatField(default=0)
    height = models.FloatField(default=0)
    weight = models.FloatField(default=0)
    Pregnancies = models.FloatField(default=0)
    Glucose = models.FloatField(default=0)
    BloodPressure = models.FloatField(default=0)
    SkinThickness = models.FloatField(default=0)
    Insulin = models.FloatField(default=0)
 

    
class DietRecommendation(models.Model):
    gender = models.CharField(max_length=10)
    activity = models.IntegerField()
    weight_loss_plan = models.CharField(max_length=20)
    diet = models.TextField()


class User(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    dob = models.DateField()
    phone_number = models.CharField(max_length=20)
    address = models.CharField(max_length=255)


class PredictionForm(forms.Form):
    Pregnancies = forms.FloatField()
    Glucose = forms.FloatField()
    BloodPressure = forms.FloatField()
    SkinThickness = forms.FloatField()
    Insulin = forms.FloatField()
    Height = forms.FloatField()
    Age = forms.FloatField()
    Weight = forms.FloatField()