from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse

from .models.user import User
from .services import userSVC

from django.core import serializers
import json

from .utils import Utils


def index(request):
    list = userSVC.userList(20, '-id')
    return render(request, "index.html", Utils.response(1, "", list))


def signIn(request):
    email = request.POST.get("email")
    password = request.POST.get("password")
    if userSVC.checkUser(email):
        return JsonResponse(Utils.response(0, "already exists"))
    else:
        return JsonResponse(Utils.response(1, "available"))
    # print(json.dumps(data, separators=(',', ':')))
    # print(JsonResponse(data))
    # return HttpResponse(json.dumps(data), "application/json")


def joinUser(request):
    email = request.POST.get("email")


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


def join(request):
    context = {}
    return render(request, "join.html", context)
