# Feature-Pyramid-Recurrent-Residual-U-Net-for-Metallographic-Image-Segmentation
In this repository a novel U-Net architecture s being proposed which contains Feature Pyramid in the decoder part and has recurrent residual block along with squeeze and excitation channel for efficient feature extraction.

![latest_FPN_R2UNET drawio](https://github.com/sammajum706/Feature-Pyramid-Recurrent-Residual-U-Net-for-Metallographic-Image-Segmentation/assets/102480479/ac3f6645-6f40-4437-aa3a-508accdafd91)

# Dataset Link
Download the [MetalDAM](https://github.com/ari-dasci/OD-MetalDAM) dataset.

# Code Instructions

For running the code the main.py is sufficient to have the desired results:
```
python main.py --images_path <image path> --masks_path <mask path>
```
The arguments are as follows:

`--images_path` : The folder directory which contains the images.

`--masks_path` : The folder directory which contains the masks.

`--epochs` : The number of epochs used for training the novel U-Net model.

`--batch` : The batch size applied on the dataset for training the model.

`--n_splits` : The number of folds required for training.

`--lr` : The learning rate required for training the model.

`--is_augment` : Takes the boolean value from the user that whether data augmentation is required or not.

`--model_directory` : The directory where the model is stored after training.

`--show` : Takes the boolean value for showing the comparison among the original, ground-truth and predicted images for the test dataset.

To know about the details of the arguments required for running the main.py file, run the following code : 
```
python main.py --help
```

Reuired directory structure :
```
+-- data
|   +-- images
|   |   +--image00
|   |   +--image01
|   |   +--image02
|   |   ...
|   +-- masks
|   |   +--mask00
|   |   +--mask01
|   |   +--mask02
|   |   ...
+-- main.py
```


