from django.http import HttpResponse
from .utils import Utils
from django.shortcuts import render
from django.http import JsonResponse
from django.core import serializers

from .services import userSVC
from django.forms.models import model_to_dict


def index(request):
    list = userSVC.userList(20, '-id')
    jStr = serializers.serialize("json", list)
    data = {"list": list, "jStr": jStr}
    return render(request, "index.html", Utils.response(1, "", data))


def signIn(request):
    email = request.POST.get("email")
    password = request.POST.get("password")

    ret = userSVC.userLogin(email, password)
    if ret:
        return JsonResponse(Utils.response(1, "succ", serializers.serialize("json", ret)))
    else:
        return JsonResponse(Utils.response(-1, "일치하는 계정이 없습니다."))


def joinUser(request):
    email = request.POST.get("email")
    password = request.POST.get("password")
    name = request.POST.get("name")
    nick = request.POST.get("nick")
    if userSVC.checkUser(email):
        return JsonResponse(Utils.response(-1, "already exists"))

    userSVC.userJoin(email, password, name, nick)
    return JsonResponse(Utils.response(1, "succ"))


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
