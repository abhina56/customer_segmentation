from django.db import models


class User(models.Model):
    name = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=255)
    purpose = models.CharField(max_length=255)
    gender = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    dob = models.DateField()
    verification_token = models.CharField(max_length=255)
    username = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    Verified = models.CharField(max_length=255)



class Meta:
    db_table = 'user'  # Specify the table name as 'user' in your database



