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


def detail(request, id):
    return HttpResponse("test %s" % id)

