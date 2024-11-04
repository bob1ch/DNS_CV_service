from minio import Minio
from minio.error import S3Error
import os

def upload_folder_to_minio(folder_path, bucket_name, client):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            object_name = os.path.relpath(file_path, folder_path)
            try:
                client.fput_object(bucket_name, object_name, file_path)
                print(f"'{file_path}' successfully uploaded as '{object_name}'")
            except S3Error as err:
                print(f"Error occurred while uploading {file_path}: {err}")

def push_data(ds_path, client):
    runs_dir = 'runs/detect/'
    model_path = os.path.join(runs_dir, sorted(os.listdir(runs_dir))[-1], 'weights/')

    #dataset
    bucket_name = "datasets-bucket"
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        upload_folder_to_minio(ds_path, bucket_name, client)
        print(f'Dataset {bucket_name} uploaded to bucket')
    else:
        print(f"Bucket '{bucket_name}' уже существует")

    #model
    bucket_name = "model-bucket" 
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
    else:
        print(f"Bucket '{bucket_name}' уже существует")
    for model in os.listdir(model_path):
        client.fput_object(bucket_name, model, os.path.join(model_path, model))
        print(f"Uploaded model to {bucket_name}/{model} in MinIO")