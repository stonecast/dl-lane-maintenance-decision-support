
## for runserver
``` 
conda activate torch-gpu

cd C:\Users\LWS\Documents\education\University\Tech_University_of_Korea\high_way_image_segmentation\web_ui_making\lane_detector

python manage.py runserver
```



## For migration
```
python manage.py makemigrations ui

python manage.py migrate
```

## for create account
```
python manage.py createsuperuser
python manage.py changepassword <username>
```
admin's id = admin / dldnjstjr123


apikey_its = "71673da9ec024f2198d3735cb6a4e5e1"
apikey_tmap = "MKaDMDXsqV9vbj4fklbqfU0E7nGvxvywkWG0JU00"

## for hosting
``` for ngrok
ngrok config add-authtoken 35kXCMUKmioXDEzkqZWVMR94tex_2u7A5qMDtCuzpnjrbX65p
ngrok http 8000
```
