from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class PatientRegistrationForm(UserCreationForm):
    dob = forms.DateField(label='Date of Birth', required=True, input_formats=['%d/%m/%Y'])
    phone_number = forms.CharField(label='Phone Number', required=True)
    address = forms.CharField(label='Address', required=True)

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'dob', 'phone_number', 'address']

    def clean_dob(self):
        dob = self.cleaned_data.get('dob')

        return dob

