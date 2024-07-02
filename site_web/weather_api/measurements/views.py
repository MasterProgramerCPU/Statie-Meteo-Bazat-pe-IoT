from rest_framework import viewsets
from .models import Masuratoare, Temperatura, Presiune, Ceata, Anemometru, Umiditate, Lumina, Predictie
from .serializers import (MasuratoareSerializer, TemperaturaSerializer, PresiuneSerializer, 
                          CeataSerializer, AnemometruSerializer, UmiditateSerializer, LuminaSerializer, PredictieSerializer)
from rest_framework.response import Response
from rest_framework.decorators import api_view

class MasuratoareViewSet(viewsets.ModelViewSet):
    queryset = Masuratoare.objects.all()
    serializer_class = MasuratoareSerializer

class TemperaturaViewSet(viewsets.ModelViewSet):
    queryset = Temperatura.objects.all()
    serializer_class = TemperaturaSerializer

class PresiuneViewSet(viewsets.ModelViewSet):
    queryset = Presiune.objects.all()
    serializer_class = PresiuneSerializer

class CeataViewSet(viewsets.ModelViewSet):
    queryset = Ceata.objects.all()
    serializer_class = CeataSerializer

class AnemometruViewSet(viewsets.ModelViewSet):
    queryset = Anemometru.objects.all()
    serializer_class = AnemometruSerializer

class UmiditateViewSet(viewsets.ModelViewSet):
    queryset = Umiditate.objects.all()
    serializer_class = UmiditateSerializer

class LuminaViewSet(viewsets.ModelViewSet):
    queryset = Lumina.objects.all()
    serializer_class = LuminaSerializer

class PredictieViewSet(viewsets.ModelViewSet):
    queryset = Predictie.objects.all()
    serializer_class = PredictieSerializer
