import django
from django.core import serializers
from django.core.serializers.json import DjangoJSONEncoder
from django.db import models
import json


class User(models.Model):
    email = models.CharField(max_length=32)
    password = models.CharField(max_length=32)
    userType = models.IntegerField()
    name = models.CharField(max_length=32)
    nick = models.CharField(max_length=32)
    sex = models.IntegerField()
    bio = models.CharField(max_length=512)
    career = models.TextField()
    accessDate = models.DateTimeField()
    regDate = models.DateTimeField()
    status = models.IntegerField()

    class Meta:
        db_table = "tblUser"

    def __str__(self):
        return "{} , {}, {}".format(self.id, self.email, self.name)
        # return json.dumps(self, cls=DjangoJSONEncoder)
        # return serializers.serialize("json", self)
        # print(json.dumps(self))
        # return json.dumps(self)
