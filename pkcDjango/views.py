from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse

from .models.user import User
from .services import userSVC

from django.core import serializers
import json

def index(request):
    list = userSVC.userList(20, '-id')
    return render(request, "index.html", {'list': list, 'jStr': serializers.serialize("json", list)})


def signIn(request):
    email = request.POST.get("email")
    password = request.POST.get("password")
    print(email)
    print(password)
    data = {}
    if userSVC.checkUser(email):
        data["returnCode"] = 0
        data["returnMessage"] = "already exists"
        # data["returnMessage"] = "already exists"
    else:
        data["returnCode"] = 1
        data["returnMessage"] = "available"
        # data["returnMessage"] = "available"
    print(json.dumps(data, separators=(',', ':')))
    print(JsonResponse(data))
    return JsonResponse(data)
    # return HttpResponse(json.dumps(data), "application/json")

def about(request):
    context = {}
    return render(request, "about.html", context)


def ministries(request):
    context = {}
    return render(request, "ministries.html", context)


def sermons(request):
    context = {}
    return render(request, "sermons.html", context)


def events(request):
    context = {}
    return render(request, "events.html", context)


def blog(request):
    context = {}
    return render(request, "blog.html", context)


def contact(request):
    context = {}
    return render(request, "contact.html", context)


def login(request):
    context = {}
    return render(request, "login.html", context)
