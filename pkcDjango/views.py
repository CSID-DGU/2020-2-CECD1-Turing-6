from django.contrib.auth import authenticate
from django.http import HttpResponse
from .utils import Utils
from django.shortcuts import render
from django.http import JsonResponse
from django.core import serializers
from django.contrib.auth import login, logout
from .services import userSVC, NeuralNet
from django.forms.models import model_to_dict


def index(request):
    list = userSVC.userList(20, '-id')
    jStr = serializers.serialize("json", list)
    data = {"list": list, "jStr": jStr, "user": request.user}
    return render(request, "index.html", Utils.response(1, "", data))


def signIn(request):
    email = request.POST.get("email")
    password = request.POST.get("password")

    user = authenticate(email=email, password=password)
    if user is not None:
        login(request, user)
        return JsonResponse(Utils.response(1, "succ"))
    else:
        return JsonResponse(Utils.response(-1, "일치하는 계정이 없습니다."))


def signOut(request):
    logout(request)
    return JsonResponse(Utils.response(1, "로그아웃 되었습니다."))


def joinUser(request):
    email = request.POST.get("email")
    password = request.POST.get("password")
    phone = request.POST.get("phone")
    name = request.POST.get("name")
    nick = request.POST.get("nick")
    if userSVC.checkUser(email):
        return JsonResponse(Utils.response(-1, "already exists"))

    user = userSVC.userJoin(email, password, name, nick, phone)
    login(request, user)
    return JsonResponse(Utils.response(1, "가입되었습니다."))


def upload(request):
    if request.method == 'POST':
        file = userSVC.uploadFile(request.POST, request.user, request.FILES['img'])
        if file:
            # NeuralNet.img_seg(file)
            userSVC.addAnalyze(request.user.id, file.id, 0)
            return JsonResponse(Utils.response(1, "succ"))
        else:
            return JsonResponse(Utils.response(2, "파일업로드 실패"))
    else:
        return JsonResponse(Utils.response(-1, "통신 방식이 잘못되었습니다."))


def faq(request):
    list = userSVC.faqList()
    jStr = serializers.serialize("json", list)
    data = {"list": list, "jStr": jStr, "user": request.user}
    print(data)
    return render(request, "faq.html", Utils.response(1, "", data))


def history(request):
    list = userSVC.historyList(20, '-id')
    jStr = serializers.serialize("json", list)
    data = {"list": list, "jStr": jStr, "user": request.user}
    return render(request, "history.html", Utils.response(1, "", data))


def about(request):
    context = {}
    return render(request, "about.html", context)


def analyzer(request):
    context = {}
    return render(request, "analyzer.html", context)


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


def loginUser(request):
    context = {}
    return render(request, "login.html", context)


def join(request):
    context = {}
    return render(request, "join.html", context)
