# This is the back end repository for 3D Dollface

## installatio

* Before installing please Download these files and copy them to data/
	- data/FLAME_albedo_from_BFM.npz
	- data/deca_model.tar
	- data/generic_model.pkl
	- data/Face_DeepLabV3+_TimmRegnety002.pth

1- Make sure you install python 3.11
```
	$ python -V
```


2- Create Virtual enviroment
```
	$ pyhton -m venv vnev
```
3- Activate your enviroment
```
	$ source venv/bin/activate

```
4- Make sure you computer support Cudas
```
        $ nvidia-smi

```
If your os Detect Cuda then run that
```
        $ cd decalib/utils/rasterizer
        $ python setup.py build_ext -i
```

5- Run this commands
```
	$ python manage.py makemigrations
	$ python manage.py migrate
```

6- After that you just start program
```
	$ python manage.py runserver
```

7- Please check your server currently running or not in https://127.0.0.1:8000
