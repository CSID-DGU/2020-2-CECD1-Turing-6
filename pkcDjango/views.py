from django.http import HttpResponse
from django.shortcuts import render
from .models.user import User


def index(request):
    list = User.objects.order_by('-regDate')[:20]
    # output = ', '.join([str(item.id) for item in list])
    context = {'list': list}
    # return HttpResponse(output)
    return render(request, "index.html", context)


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
