from rest_framework.mixins import ListModelMixin , CreateModelMixin 
from rest_framework.generics import GenericAPIView 
from .models import Object3DModel
from .serializers import Object3DSerializer

from rest_framework.response import Response
from uuid import uuid4
from rest_framework import status
from .utils import  create_3d

class HomeView(ListModelMixin , CreateModelMixin, GenericAPIView):
    queryset = Object3DModel.objects.all()
    serializer_class = Object3DSerializer

    def get(self, request, *args, **kwargs):
        print(f'args= {args} kwargs = {kwargs} {request.GET}')
        return self.list(request, *args, **kwargs)
    
    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)
       
    def create(self, request, *args, **kwargs):
        instance = Object3DModel.objects.create(uuid = uuid4())
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        if serializer.is_valid(raise_exception=True):
            self.perform_create(serializer)
            try:
                create_3d(instance)
            except:
                return Response(data='Sorry, we cant Create your 3D object right now, please try again or contact you developer')

        else:
            instance.delete()
        
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)



