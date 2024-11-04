## 1

```
pip install -r requirements.txt
```

### 1.1

```
mlflow server --backend-store-uri runs/mlflow
```

### 1.2

запустить юпитер ноутбук

### results

Юпитер ноутбук содержит преобразование датасета в формат йоло, некоторый анализ и блок обучения yolov8n

по адрессу http://127.0.0.1:5000 можно посмотреть артефакты обучения

## 2

## 2.1

```
systemctl start docker.service
```

## 2.2
```
sudo docker run \
   -p 9000:9000 \
   -p 9001:9001 \
   --name minio \
   -v /data \
   -e "MINIO_ROOT_USER=test" \
   -e "MINIO_ROOT_PASSWORD=test123456" \
   quay.io/minio/minio server /data --console-address ":9001"
```

## 2.3

```
python train.py --dataset=datasets-bucket --model=best.pt --epochs=1
```

## 3

```
bentoml serve .
```

```
python client.py 
```

