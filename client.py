import bentoml
import time
from minio import Minio
from modules import push_to_minio


client = bentoml.SyncHTTPClient('http://localhost:3000')

#мало ли в minio не попал датасет
client_minio = Minio(
    "localhost:9000",
    access_key="test",
    secret_key="test123456",
    secure=False
)
push_to_minio.push_data('datasets/dataset_notebook', client_minio)

#начинаем стресс тест))
tasks = []
for i in range(200):
    tasks.append(client.predict.submit(bucket="datasets-bucket", path=f'test/images/{i}.jpg'))
    print("Task submitted, ID:", tasks[-1].id)

while True:
    succ_counter = 0
    for task in tasks:
        cur_stat = task.get_status().value
        print(f'task_id: {task.id} status: {cur_stat}')
        if cur_stat == 'success':
            succ_counter += 1
    print('='*50)
    if succ_counter == len(tasks):
        break

# while (status := task.get_status()).value != 'success':
#     if status.value == 'failure':
#         print("The task run failed.")
#     else:
#         print("The task is still running.")
#     time.sleep(1)
# print("The task runs successfully. The result is", task.get())