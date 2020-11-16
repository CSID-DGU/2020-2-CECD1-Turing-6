# from pkcDjango.models.user import User
from pkcDjango.models import User
from pkcDjango.models import File
from pkcDjango.models import Faq
from pkcDjango.utils import Utils
from pkcDjango.models import Analyze


def userList(limit=None, order='-id', **filters):
    # return User.objects.raw('SELECT * FROM tblUser WHERE status = 1 ORDER BY regDate DESC')
    if limit:
        return User.objects.filter(**filters).order_by(order)[:limit]
    return User.objects.filter(**filters).order_by(order)


def faqList(limit=None, order="-id", **filters):
    if limit:
        return Faq.objects.filter(**filters).order_by(order)[:limit]
    return Faq.objects.filter(**filters).order_by(order)


def checkUser(email):
    return User.objects.filter(email__exact=email).exists()


def userLogin(email, password):
    user = User.objects.filter(email__exact=email, password__exact=Utils.AESCipher().encrypt(password))[:1]
    return user


def userJoin(email, password, name, nick, phone):
    User.objects.create_user(email, password, name=name, nick=nick, phone=phone)
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
    file.save()
    return File.objects.latest("id")


def addAnalyze(userId, title, fileId, resFileId):
    analyze = Analyze()
    analyze.userId = userId
    analyze.title = title
    analyze.fileId = fileId
    analyze.resFileId = resFileId
    analyze.save()
    return analyze


def historyList(limit=None, order='-id', query=None):
    whereStmt = "status = 1 "
    if query:
        whereStmt += f"AND title LIKE'%%{query}%%'"

    print(whereStmt)
    return Analyze.objects.raw(
        '''
        SELECT 
            *,
            (SELECT originName FROM tblFile WHERE id = A.fileId) AS originName,
            (SELECT path FROM tblFile WHERE id = A.fileId) AS originPath,
            (SELECT originName FROM tblFile WHERE id = A.resFileId) AS resName,
            (SELECT path FROM tblFile WHERE id = A.resFileId) AS resPath 
        FROM tblAnalyze A
        WHERE {}
        ORDER BY regDate DESC
        '''.format(whereStmt)
    )
