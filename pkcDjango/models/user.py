import json
import django
from django.db import models
from django.utils import timezone
from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.utils.translation import ugettext_lazy as _, ugettext
from django.urls.base import reverse
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.base import ModelBase

from django.db.models.signals import post_migrate
from django.contrib.auth.models import Permission

import datetime
import decimal

from django.core import serializers


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
        # return serializers.serialize("json", self)
