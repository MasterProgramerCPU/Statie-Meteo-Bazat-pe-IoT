# Generated by Django 5.0.6 on 2024-06-25 17:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('measurements', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Predictie',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dataMasuratoare', models.CharField(max_length=100)),
                ('timpMasuratoare', models.IntegerField()),
                ('valoare_temperatura', models.IntegerField()),
                ('valoare_presiune', models.IntegerField()),
                ('valoare_ceata', models.IntegerField()),
                ('valoare_anemometru', models.IntegerField()),
                ('valoare_umiditate', models.IntegerField()),
                ('valoare_Lumina', models.IntegerField()),
            ],
        ),
    ]
