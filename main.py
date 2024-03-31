from input import*
from fpn_r2unet import*
from segmented_mask import*
from training import*
from evaluation import*
from augment_data import*
import argparse
import os




parser = argparse.ArgumentParser()

parser.add_argument('--images_path', type=str, default = './',
                    help='Path where the images folder is stored')

parser.add_argument('--masks_path', type=str, default = './',
                    help='Path where the masks folder is stored')

parser.add_argument('--epochs', type=int, default = 150,
                    help='Number of Epochs for training')

parser.add_argument('--batch', type=int, default = 4,
                    help='Batch Size for Mini Batch Training')

parser.add_argument('--n_splits', type=int, default = 6,
                    help='Number of folds for training')

parser.add_argument('--lr', type=float, default = 1e-3,
                    help='Learning rate for training')

parser.add_argument('--shape',type=tuple,default=(224,224),
                    help='Shape of the image and the mask')

parser.add_argument("--is_augment",type=bool,default=False,
                    help="Takes the call whether to augment the dataset or not")

parser.add_argument("--model_directory",type=str,deafult="/",
                    help="Folder where the trained model is saved")

parser.add_argument('--show', type=bool, default = False,
                    help='Showing the comparison among original, ground-truth and predicted images for the test dataset')

args = parser.parse_args()

image_path=args.images_path
mask_path=args.mask_path
model_path=args.model_directory
n_folds=args.n_spilts
require_augment=args.is_augment
img_shape=args.shape
x=os.listdir(image_path)
y=os.listdir(mask_path)

learning_rate=args.lr
batch_size=args.batch
epochs=args.epochs

if require_augment== False:
    augment_data(image_path, mask_path)

for i in range(n_folds):

    X_train,y_train,X_val,y_val,X_test,y_test,y_test2= kfold(x,y,i,image_path,mask_path,img_shape)
    model= Feature_Pyr_R2_unet(input_shape=(img_shape[0],img_shape[1],1))
    train_model(model, model_path+"/saved_model_fold"+str(i+1)+".hdf5",learning_rate,X_train,y_train,X_val,y_val,batch_size,epochs)
    mean_iou_score(model=model,X_test=X_test,y_test=y_test,dsize=img_shape)
    rest_metrics(model,X_test,y_test)

    show=args.show

    if show== True:
        for i in range(X_test.shape([0])):
            model_prediction(model, X_test, y_test,i)
            
