from pkcDjango.models.user import User


def userList(limit=None, order='-id', **filters):
    # return User.objects.raw('SELECT * FROM tblUser WHERE status = 1 ORDER BY regDate DESC')
    if limit:
        return User.objects.filter(**filters).order_by(order)[:limit]
    return User.objects.filter(**filters).order_by(order)


def checkUser(email):
    return User.objects.filter(email__exact=email).exists()
