
#------------------------------------------------------------------------------------------------------------------------------------
#                                                        imports
#------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from PIL import Image
    
#------------------------------------------------------------------------------------------------------------------------------------
#                                                        shapes
#------------------------------------------------------------------------------------------------------------------------------------

# (w,h) npa => (w,h,1) npa
def go3D(npa):
    shape=np.shape(npa)
    if len(shape)==2:(n0,n1)=shape
    if len(shape)==3:(n0,n1,_)=shape
    new_npa=np.zeros((n0,n1,3),dtype=np.uint8)
    for i in range(3):
        if len(shape)==2:new_npa[:,:,i]=npa
        if len(shape)>2:new_npa[:,:,i]=npa[:,:,0]
    return new_npa
    
# 2D npa => 3D image
def get_image_from_npa(npa):
    shape=np.shape(npa)
    if len(shape)==2:
        npa=go3D(npa)
        image=Image.fromarray(npa, 'RGB')
    if len(shape)>2:
        if shape[2]==1:image = Image.fromarray(go3D(npa), 'RGB')
        if shape[2]==3:image = Image.fromarray(npa, 'RGB')
        if shape[2]==4:image = Image.fromarray(npa, 'RGBA')
    return image

# resize 2D npa / 3D image
def resize_image(npa,new_width):
    img=get_image_from_npa(npa)
    wpercent = (new_width/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((new_width,hsize), Image.ANTIALIAS)
    npa=np.asanyarray(img)
    return npa

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        image <=> npa
#------------------------------------------------------------------------------------------------------------------------------------

def save_npa_as_image(npa,p):
    get_image_from_npa(npa).save(p)

def get_image_from_npa(npa):
    if np.shape(npa)[2] == 3: image = Image.fromarray(npa, 'RGB')
    if np.shape(npa)[2] == 4: image = Image.fromarray(npa, 'RGBA')
    return image

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        display
#------------------------------------------------------------------------------------------------------------------------------------

def add_shape(npa,top,left,low,right,r,g,b):
    for i in range(len(npa)):
        for j in range(len(npa[0])):
            if top<=i<=low and left<=j<=right:
                npa[i,j,0] = r
                npa[i,j,1] = g
                npa[i,j,2] = b
    return npa

def draw_line(num_line,p_t,r,g,b,a):
    t = copy.copy(p_t)
    for num_column in range(len(t[0])):
        pixel = t[num_line,num_column]
        pixel[0] = r
        pixel[1] = g
        pixel[2] = b
        pixel[3] = a
    return t

def draw_column(num_column,p_t,r,g,b,a):
    t = copy.copy(p_t)
    for num_line in range(len(t)):
        pixel = t[num_line,num_column]
        pixel[0] = r
        pixel[1] = g
        pixel[2] = b
        pixel[3] = a
    return t

def add_point(npa,i_line,i_column):
    npa_draw=draw_column(i_column,npa,0,255,0,255)
    npa_draw=draw_line(i_line,npa_draw,0,255,0,255)
    img=get_image_from_npa(npa_draw)
    return img

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        à trier
#------------------------------------------------------------------------------------------------------------------------------------

# post-process mnist + cifar
def post_process_generated_image_1(c,image):
    image = np.reshape(image, (10, 10, 32, 32, 3))
    image = np.transpose(image, (0, 2, 1, 3, 4))
    image = np.reshape(image, (10 * 32, 10 * 32, 3))
    image = 255 * (image + 1) / 2
    image = image.astype("uint8")
    return image

def post_process_generated_image_2(c,image):
    n=10 #1
    #image = generator.predict(np.random.normal(size=(n * n,) + noise_size))
    image = np.reshape(image, (n, n, 28, 28, 1))
    image = np.transpose(image, (0, 2, 1, 3, 4))
    image = np.reshape(image, (n * 28, n * 28, 1))
    image = 255 * (image + 1) / 2
    image = image.astype("uint8")
    image=get_image_from_npa(image)
    return image

# ip
def post_process_generated_image_3(c,image):
    n=10 #1
    image = np.reshape(image, (n, n, 3, 5, 1))
    image = np.transpose(image, (0, 2, 1, 3, 4))
    image = np.reshape(image, (n * 3, n * 5, 1))
    image = 600 * (image + 1) / 2
    image = image.astype("uint8")
    #image=get_image_from_npa(image)
    return image