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


