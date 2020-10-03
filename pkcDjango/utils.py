from django.core import serializers
import json

class Utils:


    @staticmethod
    def response(returnCode, returnMessage, returnData=None):
        return {'returnCode': returnCode, 'returnMessage': returnMessage, 'returnData': returnData}