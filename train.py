import argparse
from minio import Minio
from modules import yolo_train, push_to_minio

MODELS_PATH = 'models'
MODELS_BUCKET = 'model-bucket'
DATASETS_PATH = 'datasets'

client = Minio(
    "localhost:9000",
    access_key="test",
    secret_key="test123456",
    secure=False
)
push_to_minio.push_data('datasets/dataset_notebook', client)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="укажите адресс датасета в minIO")
	parser.add_argument("--model", help="укажите название модели в minIO")
	parser.add_argument("--epochs", help="укажите кол-во эпох")
	args = parser.parse_args()
	return args.dataset, args.model, args.epochs

if __name__ == "__main__":
	bucket, model, epochs = get_args()

	model_path = yolo_train.get_model(MODELS_BUCKET, model, MODELS_PATH, client)
	dataset_path = yolo_train.get_dataset(DATASETS_PATH, bucket, client)
	yolo_train.train_model(dataset_path, model_path, epochs=int(epochs))
	push_to_minio.push_data('datasets/dataset_notebook', client)