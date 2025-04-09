import os
import argparse
from ultralytics import YOLO
from YamlCreater import CreateYamlFile

def RunYoloModel(model_name, dataset_dir, img_size, batch_size, epochs):
    if model_name == "yolov5":
        model = YOLO("yolov5nu.pt")
    elif model_name == "yolov8":
        model = YOLO("yolov8n.pt")
    elif model_name == "yolov11":
        model = YOLO("yolo11n.pt")
    else:
        exit(1)
    
    yaml_path = CreateYamlFile(dataset_dir)
    results_name = f'{model_name}_{os.path.dirname(dataset_dir)}_results'

    print(f"*** Training started...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=results_name,
        name='train',
        exist_ok=True
    )
    print(f"*** Training finished, results saved in {results_name}/train")
    model.val()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   type=str, required=True)
    parser.add_argument('--img_size',    type=int, default=640)
    parser.add_argument('--model',       type=str, required=True, choices=['yolov5', 'yolov8', 'yolov11', 'all'])
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--epochs',      type=int, default=50)
    arguments = parser.parse_args()
    
    if arguments.model == 'all':
        models = ['yolov5', 'yolov8', 'yolov11']
    else:
        models = [arguments.model]
    
    for model_name in models:
        RunYoloModel(
            dataset_dir =   arguments.data_path,
            img_size =      arguments.img_size,
            model_name =    model_name,
            batch_size =    arguments.batch_size,
            epochs =        arguments.epochs,
        )

if __name__ == "__main__":
    main()