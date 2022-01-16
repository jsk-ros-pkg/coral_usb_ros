# Training Object detection model + xml format dataset

## Dataset annotation with xml file

Please follow [here](https://github.com/himlen1990/multiple-grasping-pose-learning) and annotate dataset in xml format.

## Build docker image

Then, you need to build docker image called `train-edgetpu-object-detection-xml`.

```bash
roscd coral_usb_ros/training/xml_objects
make
```

## Train model

Finally, you can train the model with your dataset.

```bash
roscd coral_usb_ros/training/xml_objects
bash ./run.sh <your_dataset_path>
```

After training, you can get trained model and label file as below:

- EdgeTPU model: `<your_dataset_path>/learn/models/output_tflite_graph_edgetpu.tflite`
- Label file: `<your_dataset_path>/learn/models/labels.txt`

### Visualize your training result with TensorBoard

You can visualize your training result with TensorBoard.

TensorBoard port is often set around `6006`.

```bash
roscd coral_usb_ros/training/xml_objects
bash ./run.sh <your_dataset_path> --port <port> tensorboard
```
