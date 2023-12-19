from rest_framework import serializers
from .models import Object3DModel

class Object3DSerializer(serializers.ModelSerializer):
    class Meta:
        model = Object3DModel
        fields = '__all__'