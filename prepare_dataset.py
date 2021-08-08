# load, split and scale the face images dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed


# load all images in a directory into memory
def load_images(path, size=(256, 512), which_direction="BtoA"):
    src_list, tar_list = list(), list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # split into blur and clear
        if which_direction == "AtoB":
            inputs, targets = pixels[:, :256], pixels[:, 256:]
        elif which_direction == "BtoA":
            inputs, targets = pixels[:, 256:], pixels[:, :256]
        else:
            raise Exception("invalid direction")
        src_list.append(inputs)
        tar_list.append(targets)
    return [asarray(src_list), asarray(tar_list)]


# dataset path
path = 'face_images/combined/train/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'face_images_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)