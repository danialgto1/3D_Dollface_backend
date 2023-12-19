# Generated by Django 4.2.7 on 2023-12-19 08:22

from django.db import migrations, models
import home.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Object3DModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('left', models.FileField(blank=True, null=True, upload_to=home.models.left_image_upload_path)),
                ('front', models.FileField(blank=True, null=True, upload_to=home.models.front_image_upload_path)),
                ('right', models.FileField(blank=True, null=True, upload_to=home.models.right_image_upload_path)),
                ('obj', models.FileField(blank=True, null=True, upload_to=home.models.obj_upload_path)),
                ('mtl', models.FileField(blank=True, null=True, upload_to=home.models.obj_upload_path)),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('uuid', models.CharField(blank=True, max_length=50, null=True)),
            ],
        ),
    ]