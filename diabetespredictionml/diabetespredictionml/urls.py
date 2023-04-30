"""diabetespredictionml URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from diabetespred import views




urlpatterns = [
    
    path("admin/", admin.site.urls),
    path("",views.home, name = "home"),
    path('diabetes_prediction/', views.diabetes_prediction, name='diabetes_prediction'),
    path('dietrecomendation/', views.diet_recomendation, name='dietrecomendation'),
    path('dataexist/', views.dataexist, name='dataexist'),
    path('modify_details/', views.modify_details, name='modify_details'),
    path('register_patient/', views.register_patient, name='register_patient'), 
    path('login/', views.login_patient, name='login_patient'),    
    path('logout/', views.logout_view, name='logout'),
]