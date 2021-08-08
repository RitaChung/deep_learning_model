from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from os import listdir
from numpy import asarray


def load_single_img(path, size=(256,256)):
    src_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        pixels = (pixels - 127.5) / 127.5
        src_list.append(pixels)
    return [asarray(src_list)]

# plot source, generated and target images
def plot_images(src_img, gen_img):
    images = vstack((src_img, gen_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated']
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, 2, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # show title
        pyplot.title(titles[i])
    pyplot.show()

path = '/Users/chia/PycharmProjects/super_revolution/blur_scanned_document/original/resized/'
[imgs] = load_single_img(path)
model = load_model('model_006440.h5')
#print(imgs.shape)

ix = randint(0, len(imgs), 1)
src_image = imgs[ix]
#print(src_image.shape)
gen_image = model.predict(src_image)
plot_images(src_image, gen_image)
