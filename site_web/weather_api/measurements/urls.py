from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import (MasuratoareViewSet, TemperaturaViewSet, PresiuneViewSet, 
                    CeataViewSet, AnemometruViewSet, UmiditateViewSet, LuminaViewSet, PredictieViewSet)

router = DefaultRouter()
router.register(r'masuratoare', MasuratoareViewSet)
router.register(r'temperatura', TemperaturaViewSet)
router.register(r'presiune', PresiuneViewSet)
router.register(r'ceata', CeataViewSet)
router.register(r'anemometru', AnemometruViewSet)
router.register(r'umiditate', UmiditateViewSet)
router.register(r'lumina', LuminaViewSet)
router.register(r'predictie', PredictieViewSet)

urlpatterns = [
    path('', include(router.urls)),
]