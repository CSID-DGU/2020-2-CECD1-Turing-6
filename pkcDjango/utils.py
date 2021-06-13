from django.core import serializers
import json
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
from django.conf import settings


class Utils:
    @staticmethod
    def response(returnCode, returnMessage, returnData=None):
        return {'returnCode': returnCode, 'returnMessage': returnMessage, 'returnData': returnData}

    class AESCipher(object):
        def __init__(self):
            key = settings.AES_KEY
            self.bs = AES.block_size
            self.key = hashlib.sha256(key.encode()).hexdigest()[:32].encode("utf-8")
            self.iv = hashlib.sha256(key.encode()).hexdigest()[:16].encode("utf-8")

        def encrypt(self, raw):
            raw = self._pad(raw)
            iv = Random.new().read(AES.block_size)
            iv = self.iv
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            return base64.b64encode(iv + cipher.encrypt(raw.encode()))

        def decrypt(self, enc):
            enc = base64.b64decode(enc)
            iv = enc[:AES.block_size]
            iv = self.iv
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

        def _pad(self, s):
            return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

        @staticmethod
        def _unpad(s):
            return s[:-ord(s[len(s) - 1:])]
