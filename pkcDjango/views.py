from django.http import HttpResponse
from django.shortcuts import render
from .models.user import User
from django.db import connection


def index(request):
    # try:
    #     cursor = connection.cursor()
    #     ins = "SELECT * FROM tblUser WHERE status = 1 ORDER BY regDate DESC"
    #     res = cursor.execute(ins)
    #     list = cursor.fetchall()
    #     connection.close()
    # except:
    #     connection.rollback()
    #     print("query failed")

    list = User.objects.raw('SELECT * FROM tblUser WHERE status = 1 ORDER BY regDate DESC')
    # for row in list:
    #     print(', '.join(
    #         ['{}: {}'.format(field, getattr(row, field))
    #          for field in ['id', 'name', 'email']]
    #     ))

    # list = User.objects.order_by('-regDate')[:20]
    # output = ', '.join([str(item.id) for item in list])
    # context = {'list': list}
    # return HttpResponse(output)
    return render(request, "index.html", {'list': list})


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
