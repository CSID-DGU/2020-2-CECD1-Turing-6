# from pkcDjango.models.user import User
from pkcDjango.models import User
from pkcDjango.models import File
from pkcDjango.utils import Utils


def userList(limit=None, order='-id', **filters):
    # return User.objects.raw('SELECT * FROM tblUser WHERE status = 1 ORDER BY regDate DESC')
    if limit:
        return User.objects.filter(**filters).order_by(order)[:limit]
    return User.objects.filter(**filters).order_by(order)


def checkUser(email):
    return User.objects.filter(email__exact=email).exists()


def userLogin(email, password):
    user = User.objects.filter(email__exact=email, password__exact=Utils.AESCipher().encrypt(password))[:1]
    return user


def userJoin(email, password, name, nick):
    User.objects.create_user(email, password, name=name, nick=nick)
    joinedUser = User.objects.get(email=email)
    return joinedUser
    # passphrase = Utils.AESCipher().encrypt(password)
    # print(passphrase)
    # print(len(passphrase))
    # user = User(email=email, password=passphrase, name=name, nick=nick, sex=1, status=1)
    # user.save()


def uploadFile(post, user, img):
    file = File()
    file.originName = img
    file.userKey = user.id
    file.path = img
    print(img)
    file.save()
    return file
