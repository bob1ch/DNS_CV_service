import os 
from ultralytics import YOLO, settings 
settings.update({"mlflow": True}) 

def get_model(bucket_name, obj_name, models_path, client):
	save_path = os.path.join(models_path, obj_name)
	client.fget_object(bucket_name, obj_name, save_path)
	print(f"Model {obj_name} downloaded")

	return save_path

def get_dataset(save_path, bucket_name, client):
	save_path = os.path.join(save_path, 'minio_'+bucket_name)
	os.makedirs(save_path, exist_ok=True)

	objects = client.list_objects(bucket_name, recursive=True)
	for obj in objects:
	    file_path = os.path.join(save_path, obj.object_name)
	    os.makedirs(os.path.dirname(file_path), exist_ok=True)
	    client.fget_object(bucket_name, obj.object_name, file_path)
	    print(f"Downloaded {file_path}")

	return save_path

def train_model(dataset_path, model_name, epochs=1): 
	dataset_path = os.path.join(dataset_path, 'dataset.yaml')
	model = YOLO(model_name)
	results = model.train(data=dataset_path, epochs=epochs)
