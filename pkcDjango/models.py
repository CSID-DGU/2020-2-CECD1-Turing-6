import os

from django.contrib.auth.base_user import BaseUserManager
from django.db import models
from django.contrib.auth.models import AbstractBaseUser


class CustomUserManager(BaseUserManager):
    def _create_user(self, email, password=None, **kwargs):
        if not email:
            raise ValueError('이메일은 필수입니다.')
        user = self.model(email=self.normalize_email(email), **kwargs)
        user.set_password(password)
        user.save(using=self._db)

    def create_user(self, email, password, **kwargs):
        """
        일반 유저 생성
        """
        kwargs.setdefault('userType', 1)
        return self._create_user(email, password, **kwargs)

    def create_superuser(self, email, password, **kwargs):
        """
        관리자 유저 생성
        """
        kwargs.setdefault('userType', 2)
        return self._create_user(email, password, **kwargs)


class User(AbstractBaseUser):
    id = models.IntegerField(
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
    phone = models.CharField(max_length=16)
    sex = models.IntegerField(default=0)
    bio = models.CharField(max_length=512)
    career = models.TextField()
    accessDate = models.DateTimeField(auto_now=True)
    regDate = models.DateTimeField(auto_now_add=True)
    status = models.IntegerField(default=1)
    USERNAME_FIELD = 'email'
    objects = CustomUserManager()

    class Meta:
        db_table = "tblUser"

    def __str__(self):
        return "{} , {}, {}".format(self.id, self.email, self.name)


class File(models.Model):
    userKey = models.IntegerField(default=0)
    id = models.IntegerField(
        primary_key=True,
        unique=True,
        editable=False,
        verbose_name="id"
    )
    originName = models.CharField(max_length=100)
    path = models.FileField(null=True, blank=True, upload_to="tempFiles/")

    class Meta:
        db_table = "tblFile"

    def fileName(self):
        return os.path.basename(self.path.name)

    def __str__(self):
        return self.path


class Faq(models.Model):
    id = models.IntegerField(
        primary_key=True,
        unique=True,
        editable=False,
        verbose_name="id"
    )
    title = models.CharField(max_length=256)
    content = models.CharField(max_length=1024)
    regDate = models.DateTimeField()

    class Meta:
        db_table = "tblFaq"


class Analyze(models.Model):
    id = models.IntegerField(
        primary_key=True,
        unique=True,
        editable=False,
        verbose_name="id"
    )
    userId = models.IntegerField(default=0)
    title = models.CharField(max_length=128)
    fileId = models.IntegerField(default=0)
    resFileId = models.IntegerField(default=0)
    status = models.IntegerField(default=1)

    originPath = models.CharField(max_length=128)
    resPath = models.CharField(max_length=128)

    class Meta:
        db_table = "tblAnalyze"
