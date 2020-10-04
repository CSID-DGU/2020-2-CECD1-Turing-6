"""pkcDjango URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name="index"),
    path('about', views.about, name="about"),
    path('ministries', views.ministries, name="ministries"),
    path('sermons', views.sermons, name="sermons"),
    path('events', views.events, name="events"),
    path('blog', views.blog, name="blog"),
    path('contact', views.contact, name="contact"),
    path('login', views.login, name="login"),
    path('join', views.join, name="join"),


    path('api/userLogin', views.signIn, name=""),
    path('api/userJoin', views.joinUser, name="")
]
