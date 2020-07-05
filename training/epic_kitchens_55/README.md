# Training Object detection model + [Epic Kitchens 55 Dataset](https://epic-kitchens.github.io/2020-55.html)

## Dataset information 

You can get the dataset information in [EPIC-KITCHENS-55](https://epic-kitchens.github.io/2020-55.html). 

## Download dataset

## Build docker image

Then, you need to build docker image called `train-epic-kitchens-55-edgetpu-object-detection`.

```bash
roscd coral_usb_ros/training/epic_kitchens_55
make
```

## Train model

Finally, you can train the model with your dataset.

```bash
roscd coral_usb_ros/training/epic_kitchens_55
bash ./run.sh <your_dataset_path> <annotation_path>
```

The dataset can be downloaded with [epic-kitchens/epic-kitchens-download-scripts](https://github.com/epic-kitchens/epic-kitchens-download-scripts).
Annotation can be found in [epic-kitchens/epic-kitchens-55-annotations](https://github.com/epic-kitchens/epic-kitchens-55-annotations).

After training, you can get trained model and label file as below;
- EdgeTPU model: `<your_dataset_path>/learn/models/output_tflite_graph_edgetpu.tflite`
- Label file: `<your_dataset_path>/learn/models/labels.txt`

### Visualize your training result with TensorBoard

You can visualize your training result with TensorBoard.

TensorBoard port is often set around `6006`.

```bash
roscd coral_usb_ros/training/epic_kitchens
bash ./run.sh <your_dataset_path> --port <port> tensorboard
```
