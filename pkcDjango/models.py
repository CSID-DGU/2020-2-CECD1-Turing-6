from django.db import models
from django.contrib.auth.models import AbstractBaseUser


class User(AbstractBaseUser):
    id = models.UUIDField(
        primary_key=True,
        unique=True,
        editable=False,
        verbose_name="id"
    )
    email = models.CharField(unique=True, max_length=32)
    password = models.CharField(max_length=256)
    userType = models.IntegerField(default=1)
    name = models.CharField(max_length=32)
    nick = models.CharField(max_length=32)
    sex = models.IntegerField()
    bio = models.CharField(max_length=512)
    career = models.TextField()
    accessDate = models.DateTimeField()
    regDate = models.DateTimeField(auto_now_add=True)
    status = models.IntegerField(default=1)
    USERNAME_FIELD = 'email'

    class Meta:
        db_table = "tblUser"

    def __str__(self):
        return "{} , {}, {}".format(self.id, self.email, self.name)
