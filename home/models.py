from django.db import models
from django.utils.timezone import now
from uuid import uuid1

def front_image_upload_path(instance , filename):
    name, extension = filename.split('.')
    return f'uploads/{now().year}/{instance.uuid}/images/front.{extension}'

def left_image_upload_path(instance , filename):
    name, extension = filename.split('.')
    return f'uploads/{now().year}/{instance.uuid}/images/left.{extension}'

def right_image_upload_path(instance , filename):
    name, extension = filename.split('.')
    return f'uploads/{now().year}/{instance.uuid}/images/right.{extension}'




def obj_upload_path(instance , filename):
    return f'uploads/{now().year}/{instance.uuid}/{filename}'

class Object3DModel(models.Model):
    left = models.FileField(upload_to=left_image_upload_path, blank=True , null=True)
    front = models.FileField(upload_to=front_image_upload_path, blank=True , null=True)
    right = models.FileField(upload_to=right_image_upload_path , blank=True , null=True)
    obj = models.FileField(upload_to=obj_upload_path , blank=True , null=True)
    mtl = models.FileField(upload_to=obj_upload_path , blank=True , null=True)
    texture = models.FileField(upload_to=obj_upload_path, blank=True , null=True)
    date = models.DateTimeField(auto_now_add=True)
    uuid = models.CharField(max_length=50 , blank = True , null=True)

