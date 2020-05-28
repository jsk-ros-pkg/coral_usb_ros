# Training Object detection model + VOC format dataset

## Dataset annotation by labelme

Please follow [here](https://jsk-recognition.readthedocs.io/en/latest/deep_learning_with_image_dataset/annotate_images_with_labelme.html) and annotate dataset in VOC format.

## Build docker image

Then, you need to build docker image called `train-edgetpu-object-detection`.

```bash
roscd coral_usb_ros/training/labelme_voc
make
```

## Train model 

Finally, you can train the model with your dataset.

```bash
roscd coral_usb_ros/training/labelme_voc
bash ./run.sh <your_dataset_path>
```
After training, you can get trained model and label file as below;
- EdgeTPU model: `<your_dataset_path>/learn/models/output_tflite_graph_edgetpu.tflite`
- Label file: `<your_dataset_path>/learn/models/labels.txt`

### Visualize your training result with TensorBoard

You can visualize your training result with TensorBoard.

TensorBoard port is often set around `6006`.

```bash
roscd coral_usb_ros/training/labelme_voc
bash ./run.sh <your_dataset_path> --port <port> tensorboard
```
