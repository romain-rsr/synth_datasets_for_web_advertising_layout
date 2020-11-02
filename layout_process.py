#------------------------------------------------------------------------------------------------------------------------------------
#                                                        public imports
#------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import random 

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        private imports
#------------------------------------------------------------------------------------------------------------------------------------

'''
import basics
reload (basics)
from basics import *
'''

import sys
from importlib import reload

p_code="/home/paintedpalms/rdrive/taff/code"
sys.path.insert(0,p_code)

from basics import *
from layout_display import *

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        object samples <=> model inputs
#------------------------------------------------------------------------------------------------------------------------------------

def vectorize_sample(sample):
    n_features=5
    n_assets=3
    npa=np.zeros(n_assets*n_features)
    i=0
    for asset in sample.assets:
        npa[i+0]=get_num_cat(asset.type)
        npa[i+1]=asset.width
        npa[i+2]=asset.height
        npa[i+3]=asset.left
        npa[i+4]=asset.top
        i+=5
    return npa

def preprocess(x,y,max_delta):
    #x=x.reshape((len(x),6))
    x,y=normalize_xy(x,y,max_delta)
    x=x.reshape((len(x),9))
    y=y.reshape((len(y),9))
    return x,y

def normalize_xy(x,y,max_delta):
    max_cat=3
    x_norm=np.zeros(np.shape(x),dtype=float)
    y_norm=np.zeros(np.shape(y),dtype=float)
    w_screen,h_screen=300,600
    n_samples=len(x)
    n_assets=len(x[0])
    for i_sample in range(n_samples):
        for i_asset in range(n_assets):
            prev=x[i_sample,i_asset,0]
            
            x_norm[i_sample,i_asset,0]=x[i_sample,i_asset,0]/w_screen/max_delta     # input width
            x_norm[i_sample,i_asset,1]=x[i_sample,i_asset,1]/h_screen/max_delta     # input height
            x_norm[i_sample,i_asset,2]=x[i_sample,i_asset,2]/max_cat                # input cat
            
            y_norm[i_sample,i_asset,0]=y[i_sample,i_asset,0]/w_screen               # output width
            y_norm[i_sample,i_asset,1]=y[i_sample,i_asset,1]/w_screen               # output left
            y_norm[i_sample,i_asset,2]=y[i_sample,i_asset,2]/h_screen               # output top
    return x_norm,y_norm

def get_num_cat(tp_str):
    tp=None
    if tp_str=="red":tp=0
    if tp_str=="green":tp=1
    if tp_str=="blue":tp=2
    if tp_str=="text":tp= 0
    if tp_str=="image":tp= 1
    if tp_str=="logo":tp= 2
    if tp_str=="cta":tp= 3

    if tp==None:print("tag3",tp_str,tp)

    return tp

# real samples => xy
def get_xy_from_samples(samples):
    n_samples=len(samples)
    n_assets=len(samples[0].assets)
    x=np.zeros((n_samples,n_assets,3),dtype=int)
    y=np.zeros((n_samples,n_assets,3),dtype=int)
    for i_sample in range(n_samples):
        for i_asset in range(n_assets):
            asset=samples[i_sample].assets[i_asset]
            x[i_sample,i_asset,0]=asset.input_width
            x[i_sample,i_asset,1]=asset.input_height
            x[i_sample,i_asset,2]=get_num_cat(asset.type)
            y[i_sample,i_asset,0]=asset.width
            y[i_sample,i_asset,1]=asset.left
            y[i_sample,i_asset,2]=asset.top
    return x,y

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        model outputs <=> object samples
#------------------------------------------------------------------------------------------------------------------------------------

def devectorize_sample(npa):
    n_features=5
    n_assets=3
    n=len(npa)
    i=0
    sample=Clay()
    sample.assets=[]
    for i_assets in range(n_assets):
        asset=Clay()
        asset.type=get_str_cat(int(np.round(npa[i+0])),option_dataset)
        asset.width=npa[i+1]
        asset.height=npa[i+2]
        asset.left=npa[i+3]
        asset.top=npa[i+4]
        asset.right=asset.left+asset.width
        asset.low=asset.top+asset.height
        sample.assets.append(asset)
        i+=5
    return sample

def postprocess(x,y,max_delta):
    #x=x.reshape((len(x_test),3,2))
    x=x.reshape((len(x_test),3,3))
    y=y.reshape((len(y_test),3,3))
    x,y=denormalize_xy(x,y,max_delta)
    return x,y

def denormalize_xy(x,y,max_delta):
    max_cat=3
    x_denorm=np.zeros(np.shape(x),dtype=int)
    y_denorm=np.zeros(np.shape(y),dtype=int)
    w_screen,h_screen=300,600
    n_samples=len(x)
    n_assets=len(x[0])
    for i_sample in range(n_samples):
        for i_asset in range(n_assets):
            x_denorm[i_sample,i_asset,0]=x[i_sample,i_asset,0]*w_screen*max_delta     # input width
            x_denorm[i_sample,i_asset,1]=x[i_sample,i_asset,1]*h_screen*max_delta     # input height
            x_denorm[i_sample,i_asset,2]=x[i_sample,i_asset,2]*max_cat                # input cat
            y_denorm[i_sample,i_asset,0]=y[i_sample,i_asset,0]*w_screen               # output width
            y_denorm[i_sample,i_asset,1]=y[i_sample,i_asset,1]*w_screen               # output left
            y_denorm[i_sample,i_asset,2]=y[i_sample,i_asset,2]*h_screen               # output top
    return x_denorm,y_denorm

def get_str_cat(tp,option_dataset):

    tp_str=None
    if option_dataset==1:
        if tp==0:tp_str= "red"
        if tp==1:tp_str= "green"
        if tp==2:tp_str= "blue"

    if option_dataset>1:
        if tp==0:tp_str= "text"
        if tp==1:tp_str= "image"
        if tp==2:tp_str= "logo"
        if tp==3:tp_str= "cta"

    #print("tag0",tp_str,tp)

    return tp_str

# xy => samples
def get_samples_from_xy(x,y,names,option_dataset):
    n_samples=len(x)
    n_assets=len(x[0])
    samples=[]
    for i_sample in range(n_samples):
        sample=Clay()
        if len(names)>0:sample.name = names[i_sample]
        sample.assets=[]
        for i_asset in range(n_assets):
            asset=Clay()
            asset.input_width=int(x[i_sample,i_asset,0])
            asset.input_height=int(x[i_sample,i_asset,1])
            asset.type=get_str_cat(x[i_sample,i_asset,2],option_dataset)
            asset.width=int(y[i_sample,i_asset,0])
            asset.left=int(y[i_sample,i_asset,1])
            asset.top=int(y[i_sample,i_asset,2])
            asset.height=int(asset.input_height/asset.input_width*asset.width)
            asset.right=int(asset.left+asset.width)
            asset.low=int(asset.top+asset.height)
            sample.assets.append(asset)
        samples.append(sample)
    return samples

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        archived samples <=> object samples
#------------------------------------------------------------------------------------------------------------------------------------

# coordinates => samples
def get_samples_from_coordinates(coordinates,option_dataset):
    samples=[]
    n_samples=np.shape(coordinates)[0]
    n_assets=np.shape(coordinates)[1]
    for i_sample in range(n_samples):
        sample = Clay()
        sample.assets=[]
        for i_asset in range(n_assets):
            asset=Clay()
            asset.input_width=coordinates[i_sample,i_asset,0]
            asset.input_height=coordinates[i_sample,i_asset,1]
            asset.width=coordinates[i_sample,i_asset,2]
            asset.height=coordinates[i_sample,i_asset,3]
            asset.left=coordinates[i_sample,i_asset,4]
            asset.top=coordinates[i_sample,i_asset,5]
            asset.type=get_str_cat(coordinates[i_sample,i_asset,6],option_dataset)
            asset.right=asset.left+asset.width
            asset.low=asset.top+asset.height
            sample.assets.append(asset)
        samples.append(sample)
    return samples

# samples => coordinates
def get_coordinates_from_samples(samples):
    n_samples=len(samples)
    n_assets=len(samples[0].assets)
    n_features=7
    coordinates = np.zeros((n_samples,n_assets,n_features),dtype=np.int64)
    for i_sample in range(n_samples):
        sample = samples[i_sample]
        for i_asset in range(n_assets):
            asset=sample.assets[i_asset]
            coordinates[i_sample,i_asset,0]=asset.input_width
            coordinates[i_sample,i_asset,1]=asset.input_height
            coordinates[i_sample,i_asset,2]=asset.width
            coordinates[i_sample,i_asset,3]=asset.height
            coordinates[i_sample,i_asset,4]=asset.left
            coordinates[i_sample,i_asset,5]=asset.top
            coordinates[i_sample,i_asset,6]=get_num_cat(asset.type)
            #coordinates[i_sample,i_asset,6]=get_num_cat(asset.type)
    return coordinates


def get_samples_from_text_file(option_exclude_mentions):
    file_path='/home/paintedpalms/rdrive/taff/data/automated_layout_real/pubs_madmix/segm3/segm.txt'
    file = open(file_path,"r")
    text = file.read()
    file.close()    
    lines = text.split('\n')
    names=[]
    samples=[]
    for line in lines:
        line = line[:-1]
        line_chunks=line.split(" ")
        sample = Clay()
        sample.name=line_chunks[0]
        sample.assets=[]
        i=1
        while i < len(line_chunks):        
            assets_chunks=line_chunks[i:i+5]
            asset=Clay()
            asset.type=assets_chunks[0]
            if option_exclude_mentions==0 or asset.type!="mentions":
                asset.left=int(assets_chunks[1])
                asset.top=int(assets_chunks[2])
                asset.right=int(assets_chunks[3])
                asset.low=int(assets_chunks[4])
                asset.width=asset.right-asset.left
                asset.height=asset.low-asset.top
                sample.assets.append(asset)
            i+=5
        samples.append(sample)
    return samples

def write_pl(file_path,pl):
    text=""
    for v in pl:text+=str(v)+'\n'
    fs = open(file_path,"w")
    fs.write(text)
    fs.close()

def read_pl(file_path):
    file = open(file_path,"r")
    text = file.read()
    file.close()    
    lines = text.split('\n')
    pl=[]
    for line in lines:
        pl.append(line)
    return pl

def get_names():
    names_path="/home/paintedpalms/rdrive/taff/data/automated_layout_real/pubs_madmix/segm3/names.txt"
    names=read_pl(names_path)
    return names
    
def save_names(samples):
    names=[]
    for sample in samples:names.append(sample.name)
    write_pl("names.txt",names)

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        process object samples
#------------------------------------------------------------------------------------------------------------------------------------

def select_real_samples(samples):
    n_assets_valid=3
    valid_samples=[]
    for sample in samples:
        sample_is_valid=True
        if len(sample.assets)!=n_assets_valid:sample_is_valid=False
        for asset in sample.assets:
            if asset.type=='mentions':
                sample_is_valid=False
        if sample_is_valid:
            valid_samples.append(sample)
    return valid_samples

def add_input_dimensions(sample,delta_random_max):
    delta_random_min=1/delta_random_max
    for asset in sample.assets:
        delta=delta_random_min+(delta_random_max-delta_random_min)*random.random()
        asset.input_width=int(asset.width*delta)
        asset.input_height=int(asset.height*delta)
    return sample

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        xy synth <=> y_gan
#------------------------------------------------------------------------------------------------------------------------------------

# get y_gan from xy (synth 2)
def get_y_gan_from_xy(x,y):
    n_samples=len(x)
    n_assets=len(x[0])
    y_gan=np.zeros((n_samples,n_assets,5))
    for i_sample in range(n_samples):
        for i_asset in range(n_assets):
            
            input_width=int(x[i_sample,i_asset,0])
            input_height=int(x[i_sample,i_asset,1])
            tp=x[i_sample,i_asset,2]
            width=int(y[i_sample,i_asset,0])
            left=int(y[i_sample,i_asset,1])
            top=int(y[i_sample,i_asset,2])
            height=int(input_height/input_width*width)
                             
            y_gan[i_sample,i_asset,0]=width
            y_gan[i_sample,i_asset,1]=height
            y_gan[i_sample,i_asset,2]=left
            y_gan[i_sample,i_asset,3]=top
            y_gan[i_sample,i_asset,4]=tp
            
    return y_gan

# get y_gan from xy (synth 1)
def get_y_gan_from_xy1(x,y,deltas):
    n_samples=len(x)
    n_assets=3
    y_gan=np.zeros((n_samples,n_assets,5))
    for i_sample in range(n_samples):
        for i_asset in range(n_assets):

            # input width + input height
            input_width=int(np.round(x[i_sample,3+5*i_asset]))
            input_height=int(np.round(x[i_sample,2+5*i_asset]))

            # type
            if x[i_sample,4+5*i_asset]==1:tp=0
            if x[i_sample,5+5*i_asset]==1:tp=1
            if x[i_sample,6+5*i_asset]==1:tp=2

            # width + height
            delta=deltas[i_sample,i_asset]
            width=int(np.round(input_width/delta))
            height=int(np.round(input_height/delta))

            # left + top
            left=int(np.round(y[i_sample,1+2*i_asset]))
            top=int(np.round(y[i_sample,0+2*i_asset]))
                                
            # y gan
            y_gan[i_sample,i_asset,0]=width
            y_gan[i_sample,i_asset,1]=height
            y_gan[i_sample,i_asset,2]=left
            y_gan[i_sample,i_asset,3]=top
            y_gan[i_sample,i_asset,4]=tp
            
    return y_gan

# get sample from y_gan
def get_samples_from_y_gan(y,names,option_dataset):
    n_samples=len(y)
    n_assets=len(y[0])
    samples=[]
    for i_sample in range(n_samples):
        sample=Clay()
        if len(names)>0:sample.name = names[i_sample]
        sample.assets=[]
        for i_asset in range(n_assets):
            asset=Clay()
            asset.width=int(np.round(y[i_sample,i_asset,0]))
            asset.height=int(np.round(y[i_sample,i_asset,1]))
            asset.left=int(np.round(y[i_sample,i_asset,2]))
            asset.top=int(np.round(y[i_sample,i_asset,3]))
            v=int(np.round((y[i_sample,i_asset,4])))
            asset.type=get_str_cat(v,option_dataset)
            asset.right=int(asset.left+asset.width)
            asset.low=int(asset.top+asset.height)
            sample.assets.append(asset)
        samples.append(sample)
    return samples

# get y_gan from samples
def get_y_gan_from_samples(samples):
    n_samples=len(samples)
    n_assets=3
    y_gan=np.zeros((n_samples,n_assets,5))
    for i_sample in range(n_samples):
        sample=samples[i_sample]
        for i_asset in range(n_assets):
            asset=sample.assets[i_asset]
            y_gan[i_sample,i_asset,0]=asset.width
            y_gan[i_sample,i_asset,1]=asset.height
            y_gan[i_sample,i_asset,2]=asset.left
            y_gan[i_sample,i_asset,3]=asset.top
            y_gan[i_sample,i_asset,4]=get_num_cat(asset.type)
    return y_gan


#------------------------------------------------------------------------------------------------------------------------------------
#                                                        y_gan normalisation
#------------------------------------------------------------------------------------------------------------------------------------

def normalize_y_gan(y_gan,mxw,mxh,n_tp):

    n_samples=y_gan.shape[0]
    n_assets=y_gan.shape[1]
    for i_sample in range(n_samples):
        for i_asset in range(n_assets):
    
            width=y_gan[i_sample,i_asset,0]
            height=y_gan[i_sample,i_asset,1]
            left=y_gan[i_sample,i_asset,2]
            top=y_gan[i_sample,i_asset,3]
            tp=y_gan[i_sample,i_asset,4]

            width=width/mxw*2-1
            height=height/mxh*2-1
            top=top/mxh*2-1
            left=left/mxw*2-1
            tp=tp/(n_tp-1)*2-1

            y_gan[i_sample,i_asset,0]=width
            y_gan[i_sample,i_asset,1]=height
            y_gan[i_sample,i_asset,2]=left
            y_gan[i_sample,i_asset,3]=top
            y_gan[i_sample,i_asset,4]=tp

    return y_gan
        
def denormalize_y_gan(y_gan,mxw,mxh,n_tp):
    
    n_samples=y_gan.shape[0]
    n_assets=y_gan.shape[1]
    for i_sample in range(n_samples):
        for i_asset in range(n_assets):
            
            width=y_gan[i_sample,i_asset,0]
            height=y_gan[i_sample,i_asset,1]
            left=y_gan[i_sample,i_asset,2]
            top=y_gan[i_sample,i_asset,3]
            tp=y_gan[i_sample,i_asset,4]

            width=int(np.round((width+1)/2*mxw))
            left=int(np.round((left+1)/2*mxw))
            height=int(np.round((height+1)/2*mxh))
            top=int(np.round((top+1)/2*mxh))
            tp=int(np.round((tp+1)/2*(n_tp-1)))

            y_gan[i_sample,i_asset,0]=width
            y_gan[i_sample,i_asset,1]=height
            y_gan[i_sample,i_asset,2]=left
            y_gan[i_sample,i_asset,3]=top
            y_gan[i_sample,i_asset,4]=tp

    return y_gan

if 0==1:
    p="/home/paintedpalms/rdrive/taff/jpnb/layout_gan_fall_2020/y_gan.npy"
    y_gan=np.load(p)
    y_gan=normalize_y_gan(y_gan,300,600,4)
    
    print(np.max(y_gan))
    print(np.min(y_gan))
    print("")
    
    y_gan=denormalize_y_gan(y_gan,300,600,4)
    
    print(np.max(y_gan))
    print(np.min(y_gan))
    print("")

