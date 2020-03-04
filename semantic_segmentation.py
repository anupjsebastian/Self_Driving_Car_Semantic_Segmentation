"""
semantic_segmentation.py

This file contains all the helper functions for the semantic segmentation project, including 
- defining the architecture of the Unet
- reading and displaying images 
- generating training batches
- one hot encoding
- additional metrics and losses (dice loss and dice coeff)
- display predictions


Some of the functions were taken from https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/ which was a useful reference for my project, 
and some of the others were based on the ideas from there but changed to suit my project.

"""




import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Conv2D, BatchNormalization, concatenate, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import numpy as np


def read_images_to_tensor(filepath, normalize = False):
    
    # Read the image as a tensor
    img = tf.io.read_file(filepath)
    output = tf.image.decode_image(img)
    
    # Normalize if required
    if normalize:
        output = (output - 128) / 128
    return output


def read_images(folderpath, split, masks = True):
    
    img_files = sorted(os.listdir(os.path.join(folderpath, split, 'images')))
    print('{} image files found in {} folder.'.format(len(img_files),os.path.join(split, 'images')))
    
    img_paths = [os.path.join(folderpath, split, 'images', file) for file in img_files]
    img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    frame_tensors = img_dataset.map(read_images_to_tensor)
    
    if masks:
        mask_files = sorted(os.listdir(os.path.join(folderpath, split, 'labels')))
        print('{} mask files found in {} folder.'.format(len(mask_files),os.path.join(split, 'labels')))
        mask_paths = [os.path.join(folderpath, split, 'labels', file) for file in mask_files]
        mask_dataset = tf.data.Dataset.from_tensor_slices(mask_paths)
        mask_tensors = mask_dataset.map(read_images_to_tensor)
        return frame_tensors, mask_tensors, len(img_files)
    
    else:
        return frame_tensors, len(img_files)
    
    
def show_images_and_labels(image_tensors, label_tensors = None, num_images_to_show = 1, masks = True):
    images = image_tensors.as_numpy_iterator()
    if masks:
        labels = label_tensors.as_numpy_iterator()
        
    
    for i in range(num_images_to_show):
        image = images.next().astype(np.uint8)
        if masks:
            label = labels.next().astype(np.uint8)
            fig = plt.figure(figsize = (15,15))
            fig.add_subplot(1,2,1)
            plt.imshow(image)
            plt.xticks([]) # Hiding axes
            plt.yticks([])
            fig.add_subplot(1,2,2)
            plt.imshow(label)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            
        else:
            plt.figure(figsize=(7,7))
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            
            
def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


def DataGenerator(directory, split, img_data_generator, mask_data_generator, colormap, random_seed = 42, batch_size = 5):
    path = os.path.join(directory, split)
    img_generator = img_data_generator.flow_from_directory(
        path, batch_size = batch_size, seed = random_seed, classes = ['images'])
    
    mask_generator = mask_data_generator.flow_from_directory(
        path, batch_size = batch_size, seed = random_seed, classes = ['labels'])
    while True: 
        images = img_generator.next()
        labels = mask_generator.next()
        
        encoded_label = [rgb_to_onehot(labels[0][x,:,:,:], colormap) for x in range(labels[0].shape[0])]
        
        yield images[0], np.asarray(encoded_label)
        
        
def unet(n_filters , bn = True, dilation_rate = 1, img_width = 256, img_height = 256):

    #Input Layer
    batch_shape=(img_width,img_height,3)
    inputs = Input(shape = batch_shape)
    print('Input Shape:', inputs)
    
    
    # Downblock 1
    conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(inputs)
    if bn:
        conv1 = BatchNormalization()(conv1)
        
    conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv1)
    if bn:
        conv1 = BatchNormalization()(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)
    
    
    # Downblock 2
    conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool1)
    if bn:
        conv2 = BatchNormalization()(conv2)
        
    conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv2)
    if bn:
        conv2 = BatchNormalization()(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)
    
    
    # Downblock 3
    conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool2)
    if bn:
        conv3 = BatchNormalization()(conv3)
        
    conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv3)
    if bn:
        conv3 = BatchNormalization()(conv3)
        
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)
    
    
    # Downblock 4
    conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool3)
    if bn:
        conv4 = BatchNormalization()(conv4)
        
    conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv4)
    if bn:
        conv4 = BatchNormalization()(conv4)
        
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)
    
    
    # Bottleneck
    conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool4)
    if bn:
        conv5 = BatchNormalization()(conv5)
        
    conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv5)
    if bn:
        conv5 = BatchNormalization()(conv5)
        
    
    # Upblock 1    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    
    conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up6)
    if bn:
        conv6 = BatchNormalization()(conv6)
        
    conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv6)
    if bn:
        conv6 = BatchNormalization()(conv6)
        
    
    # Upblock 2    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    
    conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up7)
    if bn:
        conv7 = BatchNormalization()(conv7)
        
    conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv7)
    if bn:
        conv7 = BatchNormalization()(conv7)
        
    
    # Upblock 3    
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    
    conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up8)
    if bn:
        conv8 = BatchNormalization()(conv8)
        
    conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv8)
    if bn:
        conv8 = BatchNormalization()(conv8)
        
    
    # Upblock 4    
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    
    conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up9)
    if bn:
        conv9 = BatchNormalization()(conv9)
        
    conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv9)
    if bn:
        conv9 = BatchNormalization()(conv9)
        
    
    # Output Layer
    conv10 = Conv2D(n_filters, (1, 1), activation='softmax', padding = 'same', dilation_rate = dilation_rate)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    
    return model



def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)



def display_predictions(generator, conversion_map, model):
    
    sample_images, sample_masks = next(generator)
    predictions = model.predict(sample_images)
    
    for i in range(0,np.shape(predictions)[0]):

        fig = plt.figure(figsize=(20,8))

        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(sample_images[i])
        ax1.title.set_text('Actual Image')


        ax2 = fig.add_subplot(1,3,2)
        ax2.set_title('Ground Truth labels')
        ax2.imshow(onehot_to_rgb(sample_masks[i],conversion_map))

        ax3 = fig.add_subplot(1,3,3)
        ax3.set_title('Predicted labels')
        ax3.imshow(onehot_to_rgb(predictions[i],conversion_map))

        plt.show()