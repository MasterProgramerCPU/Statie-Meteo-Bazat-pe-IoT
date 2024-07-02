from django.db import models

class Masuratoare(models.Model):
    dataMasuratoare = models.CharField(max_length = 100)
    timpMasuratoare = models.IntegerField()

class Temperatura(models.Model):
    valoarea_medie = models.IntegerField()

class Presiune(models.Model):
    valoare = models.IntegerField()

class Ceata(models.Model):
    ceata = models.IntegerField()

class Anemometru(models.Model):
    valoarea_medie = models.IntegerField()

class Umiditate(models.Model):
    valoarea_medie = models.IntegerField()

class Lumina(models.Model):
    valoarea_medie = models.IntegerField()

class Predictie(models.Model):
    dataMasuratoare = models.CharField(max_length = 100)
    timpMasuratoare = models.IntegerField()
    valoare_temperatura = models.IntegerField()
    valoare_presiune = models.IntegerField()
    valoare_ceata = models.IntegerField()
    valoare_anemometru = models.IntegerField()
    valoare_umiditate = models.IntegerField()
    valoare_Lumina = models.IntegerField()