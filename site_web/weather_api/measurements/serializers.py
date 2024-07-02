# myapp/serializers.py

from rest_framework import serializers
from .models import Masuratoare, Temperatura, Presiune, Ceata, Anemometru, Umiditate, Lumina, Predictie

class MasuratoareSerializer(serializers.ModelSerializer):
    class Meta:
        model = Masuratoare
        fields = '__all__'

class TemperaturaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Temperatura
        fields = '__all__'

class PresiuneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Presiune
        fields = '__all__'

class CeataSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ceata
        fields = '__all__'

class AnemometruSerializer(serializers.ModelSerializer):
    class Meta:
        model = Anemometru
        fields = '__all__'

class UmiditateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Umiditate
        fields = '__all__'

class LuminaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Lumina
        fields = '__all__'

class PredictieSerializer(serializers.ModelSerializer):
    class Meta:
        model = Predictie
        fields = '__all__'        
