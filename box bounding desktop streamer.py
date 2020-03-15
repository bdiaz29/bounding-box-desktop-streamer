from mss import mss
from tkinter import *
from tkinter import filedialog
import os.path
sct = mss()
import numpy as np
import cv2
import time
# example of face detection with mtcnn
from matplotlib import pyplot
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras import backend as K
# from numpy import asarray
# from mtcnn.mtcnn import MTCNN
import PIL
from tensorflow.keras.models import load_model
import keras
import tensorflow as tf
import cv2


window = Tk()
window.title("custom bounding box desktop streaming output")
window.geometry('500x500')
model=[]

file_frame = Frame(master=window)
file_frame.grid(column=0, row=0)


def limit(x1, y1, x2, y2, X, Y):
    if x1 >= X:
        x1 = X - 1

    if x2 >= X:
        x2 = X - 1

    if y1 >= Y:
        y1 = Y - 1

    if y2 >= Y:
        y2 = Y - 1

    ############
    if x1 < 0:
        x1 = 0

    if x2 < 0:
        x2 = 0

    if y1 < 0:
        y1 = 0

    if y2 < 0:
        y2 = 0

    x1 = int(min(x1, x2))
    x2 = int(max(x1, x2))
    y1 = int(min(y1, y2))
    y2 = int(max(y1, y2))

    if x1 == x2:
        x1 = 0
        x2 = X - 1

    if y1 == y2:
        y1 = 0
        y2 = Y - 1

    return x1, y1, x2, y2


def calculate_iou(target_boxes, pred_boxes):
    xA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])
    yA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])
    xB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])
    yB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])
    interArea = K.maximum(0.0, xB - xA) * K.maximum(0.0, yB - yA)
    boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)
    return mse + (1 - iou)


def iou_metric(y_true, y_pred):
    return calculate_iou(y_true, y_pred)


def new_image(img_arr, pred):
    img_shape = np.shape(img_arr)

    width = img_shape[1]
    height = img_shape[0]

    x1 = (pred[0, 0])
    y1 = (pred[0, 1])
    x2 = (pred[0, 2])
    y2 = (pred[0, 3])

    xmin = width * x1
    ymin = height * y1
    xmax = width * x2
    ymax = height * y2

    # limits put in for any unusual behavior and cast as integers
    x1, y1, x2, y2 = limit(xmin, ymin, xmax, ymax, width, height)

    im = Image.fromarray(img_arr)
    im2 = im.resize((width, height))
    im3 = np.array(im2)
    boxdrawn = np.zeros((height, width, 3))
    boxdrawn[0:width, 0:height, 0:3] = im3[0:width, 0:height, 0:3]
    boxdrawn = cv2.rectangle(boxdrawn, (x1, y1), (x2, y2), (255, 0, 0), 2)
    z = im3[y1:y2, x1:x2]
    z2 = Image.fromarray(z)
    z3 = z2.resize((width, height))
    z4 = np.array(z3)
    new_img_arr = np.zeros((height * 2, width, 3))
    new_img_arr[0:height, 0:width] = boxdrawn
    A = new_img_arr[height:height * 2, 0:width]
    B = z4
    new_img_arr[height:height * 2, 0:width] = z4
    new_img_arr = np.uint8(new_img_arr)
    return new_img_arr


def img_resize(img_arr, target):
    height = target[0]
    width = target[1]
    im0 = np.uint8(img_arr)
    im = Image.fromarray(im0)
    im2 = im.resize((height, width))
    im3 = np.array(im2)
    im3 = np.uint8(im3)
    return im3


def image_preprocess(img_arr, target):
    height = target[0]
    width = target[1]
    scaled = img_resize(img_arr, target)
    scaled = np.expand_dims(scaled, axis=0)
    new_arr = scaled / 255
    return new_arr

def just_box(img_arr, pred):
    img_shape = np.shape(img_arr)

    width = img_shape[1]
    height = img_shape[0]

    x1 = (pred[0, 0])
    y1 = (pred[0, 1])
    x2 = (pred[0, 2])
    y2 = (pred[0, 3])

    xmin = width * x1
    ymin = height * y1
    xmax = width * x2
    ymax = height * y2

    # limits put in for any unusual behavior and cast as integers
    x1, y1, x2, y2 = limit(xmin, ymin, xmax, ymax, width, height)
    #boxdrawn=np.uint8(img_arr)
    boxdrawn = np.zeros((height, width, 3))
    boxdrawn[0:width, 0:height, 0:3] = img_arr[0:width, 0:height, 0:3]
    boxdrawn=boxdrawn.astype('uint8')
    boxdrawn = cv2.rectangle(boxdrawn.astype('uint8'), (x1, y1), (x2, y2), (255, 0, 0), 2)

    return boxdrawn

def start():
    global model
    top=int(top_txt.get())
    left=int(left_txt.get())
    height=int(size_txt.get())
    width=int(size_txt.get())
    start_btn.configure(state='disabled')
    with mss() as sct:
        #initiate the grab monitor
        monitor = {"top": top, "left": left, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        sct_img = np.array(sct_img)
        #read in array channels backward to convert to rgb
        img = sct_img[:, :, [2, 1, 0]]
        img = np.uint8(img)
        while (True):
            sct_img = sct.grab(monitor)
            sct_img = np.array(sct_img)
            img_shape=np.shape(sct_img)
            H=img_shape[0]
            W=img_shape[1]
            img0=np.zeros((H,W,3))
            img0= sct_img[0:H, 0:W, [2, 1, 0]]

            img = sct_img[:, :, [2, 1, 0]]
            img = np.uint8(img)
            #pre process imahge
            screen = image_preprocess(img, [224, 224])
            pred = model.predict(screen, batch_size=1)

            box = just_box(img0, pred)
            cv2.imshow('window', cv2.cvtColor(box, cv2.COLOR_BGR2RGB))
            #close application if q is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    start_btn.configure(state='normal')
#load which H5 model to use
def load_h5():
    global model
    #load in the model
    model_source = filedialog.askopenfilename(initialdir="E:/machine learning/saved models", title="Select file",
                                              filetypes=(("h5", "*.h5"), ("all files", "*.*")))
    # read in model
    # since the model is only going to infer and not be trained
    # the model is compiled on site with standard
    # avoids conflicts with custom loss and optimizers
    model = load_model(model_source, compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    load_h5_lbl.configure(text=model_source)
    start_btn.configure(state='normal')


load_h5_btn=Button(master=file_frame,text="load h5 file",command=load_h5)
load_h5_btn.grid(column=0, row=0,padx=5, pady=5)

load_h5_lbl=Label(master=file_frame)
load_h5_lbl.grid(column=1, row=0,padx=5, pady=5)



top_lbl=Label(master=file_frame,text="Top position")
top_lbl.grid(column=0,row=1,padx=5, pady=5)

left_lbl=Label(master=file_frame,text="Left position")
left_lbl.grid(column=0,row=2,padx=5, pady=5)

width_lbl=Label(master=file_frame,text="capture screen size")
width_lbl.grid(column=0,row=3,padx=5, pady=5)



top_txt=Entry(master=file_frame)
top_txt.grid(column=1,row=1,padx=5, pady=5)
top_txt.insert(END,"0")

left_txt=Entry(master=file_frame)
left_txt.grid(column=1,row=2,padx=5, pady=5)
left_txt.insert(END,"0")

size_txt=Entry(master=file_frame)
size_txt.grid(column=1,row=3,padx=5, pady=5)
size_txt.insert(END,"224")


start_btn=Button(master=file_frame,text="start (press q to stop)",command=start)
start_btn.grid(column=0,row=5,padx=5,pady=5)
start_btn.configure(state='disabled')

window.mainloop()


