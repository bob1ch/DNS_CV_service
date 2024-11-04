import bentoml
import io
from minio import Minio
from bentoml.validators import ContentType
from pydantic import BaseModel, Field
from PIL import Image

client_minio = Minio(
    "localhost:9000",
    access_key="test",
    secret_key="test123456",
    secure=False
)

class Instance(BaseModel):
    boundingBox: str = Field(..., alias="boundingBox")
    class_: str = Field(..., alias="class")

class Response(BaseModel):
    instances: list[Instance]

@bentoml.service(resources={"gpu": 1})
class DNS_CV:
    def __init__(self):
        from ultralytics import YOLO
        client_minio.fget_object('model-bucket', 'best.pt', 'models/best.pt')
        self.model = YOLO('models/best.pt')

    @bentoml.task(route='/image/detect')
    def predict(self, bucket:str, path: str) -> Response:
        image = self.get_image(bucket, path)
        results = self.model.predict(image)[0].boxes.cpu().numpy()

        instances = [
            Instance(**{
                "boundingBox": self.convert_bbox(instance.xywh[0].tolist(), instance.orig_shape),
                "class" : str(instance.cls.astype(int).item())
                }
            )
            for instance in results
        ]

        return Response(instances=instances)

    def get_image(self,bucket, path):
        image = Image.open(io.BytesIO(client_minio.get_object(bucket, path).read()))

        return image

    def convert_bbox(self, bbox, image_shape):
        x, y, w, h = bbox
        image_width, image_height = image_shape
        x_min = x / image_width
        y_min = y / image_height
        x_max = (x + w) / image_width
        y_max = (y + h) / image_height

        return f"{x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f}"