# from django.db import models
# from django.contrib.auth.models import AbstractBaseUser
#
#
# class User(models.Model):
#     email = models.CharField(max_length=32)
#     password = models.CharField(max_length=256)
#     userType = models.IntegerField()
#     name = models.CharField(max_length=32)
#     nick = models.CharField(max_length=32)
#     sex = models.IntegerField()
#     bio = models.CharField(max_length=512)
#     career = models.TextField()
#     accessDate = models.DateTimeField()
#     regDate = models.DateTimeField()
#     status = models.IntegerField()
#
#     class Meta:
#         db_table = "tblUser"
#
#     def __str__(self):
#         return "{} , {}, {}".format(self.id, self.email, self.name)
