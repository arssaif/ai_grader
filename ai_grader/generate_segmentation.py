
import numpy
import cv2
import numpy as np
from numpy import expand_dims
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
import os
from matplotlib.patches import Rectangle
APP_ROOT= os.path.dirname(os.path.abspath(__file__))


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    
    img=cv2.imread(filename)
    img=(cv2.resize(img,(512,512))[:,:,0]-127.0)/127.0
    im_array = []
    m_array=[]
    im_array.append(img)
    test_image = np.array(im_array).reshape(len(im_array),512,512,1)


    return test_image


def ctr_calculation(h,l):
    leftl = 0
    rightl = 0
    lefth = 0
    righth = 0
    llb = True
    lrb = True
    hlb = True
    hrb = True

    print(l.shape[0],h.shape[0],l.shape[1],h.shape[1])
    for i in range(l.shape[1]):
        for j in range(l.shape[0]):
            if l[j][i] >= 1:
                if llb:
                    leftl = i
                    llb = False
            if l[j][-i] >= 1:
                if lrb:
                    rightl = -i+512
                    lrb = False

    for i in range(h.shape[1]):
        for j in range(h.shape[0]):
            if h[j][i] >= 1:
                if hlb:
                    lefth = i
                    hlb = False
            if h[j][-i] >= 1:
                if hrb:
                    righth = -i + 512
                    hrb = False
    print(leftl,rightl,lefth,righth)                
    ctr=abs(lefth-righth)/abs(leftl-rightl)
    return ctr

# draw all results
def draw_plot(filename, path, strc):
    # load the image
    #data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(filename,cmap='gray')
    pyplot.title(strc)
    ax = pyplot.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    pyplot.savefig(path,  bbox_inches='tight')
    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def segmentation(adder, newName, user_id):
    pyplot.clf()
    pyplot.cla()
    pyplot.close()
    target = os.path.join(APP_ROOT, 'static/Patient_images')
    target = "/".join([target, adder])
    print('taget',target)
    
    def dice_coef(y_true, y_pred):
        y_true_f = keras.flatten(y_true)
        y_pred_f = keras.flatten(y_pred)
        intersection = keras.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)
    modell = tf.keras.models.load_model('static/models/segmentation/cxr_reg_model.lungs.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
    modelh = tf.keras.models.load_model('static/models/segmentation/cxr_reg_model.heart.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
    modelc = tf.keras.models.load_model('static/models/segmentation/cxr_reg_model.clavicles.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
    # define the expected input shape for the model
    input_w, input_h = 512, 512
    # define our new photo
    photo_filename = target

    image = load_image_pixels(photo_filename, (input_w, input_h))
    # make prediction


    resulth = modelh.predict(image)
    resultl = modell.predict(image)
    resultc = modelc.predict(image)

    normal_arrayh = np.squeeze(resulth.astype(int))  
    normal_arrayl = np.squeeze(resultl.astype(int))  
    normal_arrayc = np.squeeze(resultc.astype(int))  

    for i in range(normal_arrayl.shape[0]):
        for j in range(normal_arrayl.shape[1]):
            if normal_arrayl[i][j]>=1:
                normal_arrayl[i][j]=120
            if normal_arrayh[i][j]>=1:
                normal_arrayh[i][j]=200
            if normal_arrayc[i][j]>=1:
                normal_arrayc[i][j]=255
    ctr=ctr_calculation(normal_arrayh,normal_arrayl)            
    final=normal_arrayl+normal_arrayh+normal_arrayc
    ctr_string="CTR Calculated:" + str(format(ctr,'.2f')) + "       CTR Normal Range 0.4-0.5"
    if not os.path.exists('static/segmentation/User' + str(user_id)):
        os.makedirs('static/segmentation/User' + str(user_id))
    path1 = 'static/segmentation/User' + str(user_id) + '/' + newName + '.jpg'
    draw_plot(final, path1,ctr_string)
