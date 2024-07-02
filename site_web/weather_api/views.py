from django.contrib import admin
from django.urls import path, re_path, include
from django.views.generic import RedirectView
from django.views.static import serve
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('measurements.urls')),
    re_path(r'^$', RedirectView.as_view(url='/static/live_data.html', permanent=False), name='live_data'),
    re_path(r'^graphs/$', RedirectView.as_view(url='/static/graphs.html', permanent=False), name='graphs'),
    re_path(r'^predictions/$', RedirectView.as_view(url='/static/predictions.html', permanent=False), name='predictions'),
    re_path(r'^table/$', RedirectView.as_view(url='/static/table.html', permanent=False), name='table'), 
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
]
