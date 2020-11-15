


# public imports
import random 
import time 
import math 
from PIL import Image
import numpy as np 
import copy 

# private imports
from layout_process import *
from image_process import *












#------------------------------------------------------------------------------------------------------------------------------------
#                                                        # real
#------------------------------------------------------------------------------------------------------------------------------------


def read_pl(file_path):
    file = open(file_path,"r")
    text = file.read()
    file.close()    
    lines = text.split('\n')
    pl=[]
    for line in lines:
        pl.append(line)
    return pl

def get_samples_from_xy_real(x,y,names):
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
            asset.type=get_str_cat(x[i_sample,i_asset,2],2)
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
#                                                        select samples
#------------------------------------------------------------------------------------------------------------------------------------

def get_special_combinations_trigram():
    special_combinaisons=[]
    special_combinaisons.append("rrr")
    special_combinaisons.append("rrb")
    special_combinaisons.append("rgb")
    special_combinaisons.append("rbr")
    special_combinaisons.append("ggr")
    special_combinaisons.append("ggg")
    special_combinaisons.append("brb")
    special_combinaisons.append("bgb")
    special_combinaisons.append("bbg")
    special_combinaisons.append("bbb")
    return special_combinaisons

def get_special_combinations():
    special_combinaisons=[]
    special_combinaisons.append(["red","red","red"])
    special_combinaisons.append(["red","red","blue"])
    special_combinaisons.append(["red","green","blue"])
    special_combinaisons.append(["red","blue","red"])
    special_combinaisons.append(["green","green","red"])
    special_combinaisons.append(["green","green","green"])
    special_combinaisons.append(["blue","red","blue"])
    special_combinaisons.append(["blue","green","blue"])
    special_combinaisons.append(["blue","blue","green"])
    special_combinaisons.append(["blue","blue","blue"])
    return special_combinaisons

# check if synth2 sample is concerned by general rules
def is_general(sample,special_combinaisons):
    ok=1
    types=[]
    for asset in sample.assets:
        types.append(asset.type)
    if types in special_combinaisons:ok=0
    return ok

def select_general_samples(samples,option_select):
    # option select 1 => general rules
    # option select 0 => special rules
    special_combinaisons=get_special_combinations()
    valid_samples=[]
    for sample in samples:
        if is_general(sample,special_combinaisons)==option_select:valid_samples.append(sample)
    return valid_samples

def check_types(samples):
    ok=1
    for sample in samples:
        for asset in sample.assets:
            if asset.type==None:ok=0
    return ok

def get_y_gan_general(y_gan,option_rules):
    option_dataset=1
    samples=get_samples_from_y_gan(y_gan,[],option_dataset)
    samples=select_general_samples(samples,option_rules)
    y_gan_general=get_y_gan_from_samples(samples)
    return y_gan_general

'''
(copy.deepcopy(sample)
'''

















#------------------------------------------------------------------------------------------------------------------------------------
#                                                        # synth 2
#------------------------------------------------------------------------------------------------------------------------------------


# data : generate background
def set_synth_background(sample):
    # set background
    if 1==1:
        spaces=get_spaces(sample)
        centers=[]
        for space in spaces:centers.append([int(300*random.random()),int((space[1]+space[2])/2),space[0]])
    npa_sample=get_npa_sample(sample)
    map_sample=get_map(sample,centers)

    # draw background
    npa_sample=get_npa_sample(sample)
    h=600
    w=300
    for i in range(h):
        for j in range(w):
            if map_sample[i,j]==1:
                if sum(npa_sample[i,j])!=300+255:
                    if npa_sample[i,j,0]==255:npa_sample[i,j,0]=150
                    if npa_sample[i,j,1]==255:npa_sample[i,j,1]=150
                    if npa_sample[i,j,2]==255:npa_sample[i,j,2]=150
                if sum(npa_sample[i,j])==300+255:
                    npa_sample[i,j,0]=0
                    npa_sample[i,j,1]=255
                    npa_sample[i,j,2]=0            
    display(get_image_from_npa(npa_sample))

def get_spaces(sample):
    h=600
    spaces=[]
    prev_low=0
    for asset in sample.assets:
        spaces.append([asset.top-prev_low,prev_low,asset.top])
        prev_low=asset.low
    spaces.append([h-prev_low,prev_low,h])
    return spaces

def get_map(sample,centers):
    w,h=300,600
    npa_map=np.zeros((h,w),dtype=int)
    npa_scores=np.zeros((h,w),dtype=float)
    scores=[]
    for line in range(h):
        for col in range(w):
            #scoreA=scoreB=scoreC=0
            score=0
            for center in centers:
                scoreA=abs(center[0]-col)#/w
                scoreB=abs(center[1]-line)#/h
                scoreC=center[2]/h
                pA=2
                pB=2
                pC=1
                v=(math.pow(scoreA,pA)+math.pow(scoreB,pB))
                if v==0:new_score=0
                if v!=0:new_score=math.pow(scoreC,pC)/math.pow(v,1/2)
                score=max(score,new_score)
            scores.append(score)
            npa_scores[line,col]=score
    scores.sort(reverse=False)
    score_thresh=scores[int(len(scores)/5)]
    score_max=max(scores)
    for line in range(h):
        for col in range(w):
            score_ratio=(npa_scores[line,col]-score_thresh)/score_max
            v=random.random()
            if v<score_ratio:npa_map[line,col]=1
    return npa_map
    
# data : generate synth I
def create_sample(p):
    n_assets=3
    sample=Clay()
    sample.assets=[]
    prev_low=0
    for i_asset in range(n_assets):
        asset=Clay()
        asset.left=50 #100#50
        asset.right=p.width-50 #-100#-50
        asset.top=prev_low+85 #+5#+85
        asset.low=asset.top+85 #+180#+85
        asset.width=asset.right-asset.left
        asset.height=asset.low-asset.top
        prev_low=asset.low
        sample.assets.append(asset)
    return sample

def shake_sample(p,sample):
    i_asset=random.randint(0,2)
    i_feature=random.randint(0,3)
    asset=sample.assets[i_asset]
    backup_left=asset.left
    backup_width=asset.width
    backup_top=asset.top
    backup_height=asset.height
    if i_asset==0:prev_low=0
    if i_asset!=0:prev_low=sample.assets[i_asset-1].low
    if i_asset==2:next_top=p.height
    if i_asset!=2:next_top=sample.assets[i_asset+1].top
    if i_feature==0:asset.left+=-50+int(random.random()*100)
    if i_feature==1:asset.right+=-50+int(random.random()*100)
    if i_feature==2:
        asset.width+=-40+int(random.random()*100)
        asset.right=asset.left+asset.width
    if i_feature==3:
        asset.height+=-40+int(random.random()*100)
        asset.low=asset.top+asset.height
    asset.right=asset.left+asset.width
    asset.low=asset.top+asset.height
    ok1=ok2=ok3=ok4=ok5=ok6=1
    if asset.left<0:ok1=0
    if asset.right>p.width:ok2=0
    if asset.top-prev_low<20:ok3=0
    if next_top-asset.low<20:ok4=0
    if asset.width<50:ok5=0
    if asset.height<50:ok6=0
    ok=True
    if ok1+ok2+ok3+ok4+ok5+ok6<6:ok=False
    if ok==False:
        asset.left=backup_left
        asset.top=backup_top
        asset.width=backup_width
        asset.height=backup_height
        asset.right=asset.left+asset.width
        asset.low=asset.top+asset.height
    return sample


# synth 2 : add categories
def add_categories(sample):
    contains_logo=False
    contains_cta=False
    i_asset=-1
    for asset in sample.assets:
        i_asset+=1
        asset.type="text"
        if asset.width>250:
            asset.type="image"
        if asset.width<150:
            v=random.random()
            if v>=0.5 and not contains_cta:
                asset.type="cta"
                contains_cta=True
            if v< 0.5 and not contains_logo and i_asset!=1:
                asset.type="logo"
                contains_logo=True
    return sample


# synth 2 : add categories
def add_categories(sample):
    contains_logo=False
    contains_cta=False
    i_asset=-1
    for asset in sample.assets:
        i_asset+=1
        asset.type="text"
        if asset.width>250:
            asset.type="image"
        if asset.width<150:
            v=random.random()
            if v>=0.5 and not contains_cta:
                asset.type="cta"
                contains_cta=True
            if v< 0.5 and not contains_logo and i_asset!=1:
                asset.type="logo"
                contains_logo=True
    return sample

#def shake_real_sample(p,sample,delta,shift):
def shake_sample2(p,sample,delta,shift):
    
    margin=0 #20
    min_width=40
    min_height=40
    #delta=10
    
    # select asset and feature to be modified
    i_asset=random.randint(0,2)
    i_feature=random.randint(0,3)
    asset=sample.assets[i_asset]
    # set backup
    backup_left=asset.left
    backup_width=asset.width
    backup_top=asset.top
    backup_height=asset.height
    # set prev low + next stop
    if i_asset==0:prev_low=0
    if i_asset!=0:prev_low=sample.assets[i_asset-1].low
    if i_asset==2:next_top=p.height
    if i_asset!=2:next_top=sample.assets[i_asset+1].top
    # modify asset feature
    if i_feature==0:asset.left+=-delta+int(random.random()*2*delta)
    if i_feature==1:asset.right+=-delta+int(random.random()*2*delta)
    if i_feature==2:
        asset.width+=-delta+int(random.random()*2*(delta))#+shift))
        #asset.width=300
        #asset.left=0
        v=random.random()
        if v>=0.5:asset.right=asset.left+asset.width
        if v< 0.5:asset.left=asset.right-asset.width
        
    if i_feature==3:
        asset.height+=-delta+int(random.random()*2*(delta+shift))
        v=random.random()
        if v>=0.5:asset.low=asset.top+asset.height
        if v< 0.5:asset.top=asset.low-asset.height
    if i_feature==4:
        asset.top+=-delta+int(random.random()*2*delta)
        asset.height==asset.low-asset.top
    if i_feature==5:
        asset.top+=-delta+int(random.random()*2*delta)
    asset.right=asset.left+asset.width
    asset.low=asset.top+asset.height
    # checks
    ok1=ok2=ok3=ok4=ok5=ok6=1
    if asset.left<0:ok1=0
    if asset.right>p.width:ok2=0
    if asset.right<asset.left:ok2=0
    if asset.low<asset.top:ok2=0
    if asset.top-prev_low<margin:ok3=0
    if next_top-asset.low<margin:ok4=0    
    if asset.width<min_width:ok5=0
    if asset.height<min_height:ok6=0
    ok=True
    if ok1+ok2+ok3+ok4+ok5+ok6<6:ok=False
    if ok==False:
        asset.left=backup_left
        asset.top=backup_top
        asset.width=backup_width
        asset.height=backup_height
        asset.right=asset.left+asset.width
        asset.low=asset.top+asset.height
    return sample

def select_synth_samples(samples):
    valid_samples=[]
    for sample in samples:
        ok=1
        for asset in sample.assets:
            if asset.height/asset.width>1.2:
                if asset.width<250:
                    ok=0
        if ok==1:valid_samples.append(copy.deepcopy(sample))
    return valid_samples


def create_synth2(n_samples):

    # synth dataset builder
    option_add_categories=1
    max_delta=1.3
    c=Clay()
    c.height=600
    c.width=300
    samples=[]
    for i_sample in range(n_samples):
        if i_sample%10000==0:print(i_sample,time.ctime())
        sample=create_sample(c)
        for k in range(200):sample=shake_sample(c,sample)
        if option_add_categories==1:sample=add_categories(sample)
        sample=add_input_dimensions(sample,max_delta)
        samples.append(sample)
    print(time.ctime())

    # select synth samples (according to geometrics)
    samples=select_synth_samples(samples)
        
    # add input dimensions on samples
    delta_random_max=2
    for sample in samples:sample=add_input_dimensions(sample,delta_random_max)
    
    # save synth samples as xy npas
    x,y=get_xy_from_samples(samples)
    np.save("x_synth2.npy",x)
    np.save("y_synth2.npy",y)





















#-------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                  # synth 1
#-------------------------------------------------------------------------------------------------------------------------------------------------------



def set_positions(params,slots,margins):
    #set slots positions
    prev_mark = 0
    for i in range(params.n_slots):
        slot = slots[i]
        next_mark = prev_mark+margins[i]
        if params.direction=='vertical':
            slot.top = next_mark
            slot.left = random.randint(0,params.screen_width-slot.width)
        if params.direction=='horizontal':
            slot.top = random.randint(0,params.screen_height-slot.height)
            slot.left = next_mark            
        slot.low = slot.top+slot.height
        slot.right = slot.left+slot.width
        dist = 20
        '''
        if slot.top < dist:
            slot.top=0
            slot.low=slot.top+slot.height
        if slot.left < dist:
            slot.left=0
            slot.right=slot.left+slot.right
        if screen_height-slot.low < dist:
            slot.low=screen_height
            slot.top=slot.low-slot.height
        if screen_width-slot.right < dist:
            slot.right=screen_width
            slot.left=slot.right-slot.width
        prev_mark=slot.low
        '''
        if params.direction=='vertical':prev_mark = slot.low
        if params.direction=='horizontal':prev_mark = slot.right
        
    return slots


def random_margin_shift(margins,params):
    
    # select slot to modify
    i=-1
    j=-1
    while i==j:
        i = random.randint(0,len(margins)-1)
        j = random.randint(0,len(margins)-1)
    mA = margins[i]
    mB = margins[j]
    
    # set width, height to add/subtract
    max_range = params.mxm-params.mnm
    m = random.randint(-max_range,max_range)
    
    # run checks
    checks = [True,True,True,True]
    mxi=mxm
    mxj=mxm
    if i!=0 and i!=len(margins)-1:mni=params.mnm
    if j!=0 and j!=len(margins)-1:mnj=params.mnm
    if i==0 or i==len(margins)-1:mni=0
    if j==0 or j==len(margins)-1:mnj=0
    if margins[i]+m<mni:checks[0]=False
    if margins[i]+m>mxi:checks[1]=False
    if margins[j]-m<mnj:checks[2]=False
    if margins[j]-m>mxj:checks[3]=False
    
    # process the adding/subtraction
    if False not in checks:
        #print("m",m)
        margins[i]+=m
        margins[j]-=m
    return margins

'''
def init_dimensions(params):
    ts=0
    tl=0
    slots=[]
    for i in range(params.n_slots):
        slot = Clay()
        slot.lw = int((params.mnlw+params.mxlw)/2) #int(params.screen_lw/7)
        slot.cw = int((params.mncw+params.mxcw)/2) #int(params.screen_cw/3)
        if params.direction=='vertical':
            slot.height = 1/1.5*slot.lw #slot.lw #1/2*slot.lw
            slot.width = 1.5*slot.cw #slot.cw #2*slot.cw
        if params.direction=='horizontal':
            slot.height = 1/1.5*slot.cw #slot.cw #1/2*slot.cw
            slot.width = 1.5*slot.lw #slot.lw #2*slot.lw
        slots.append(slot)
        ts+=slot.cw*slot.lw
        tl+=slot.lw
    return slots,tl,ts
'''

def set_constraints(params):

    # format direction
    if params.screen_height>=250/300*params.screen_width:
        params.direction = 'vertical'
        params.screen_lw = params.screen_height
        params.screen_cw = params.screen_width
    if params.screen_height<250/300*params.screen_width:
        params.direction = 'horizontal'
        params.screen_lw = params.screen_width
        params.screen_cw = params.screen_height    

    # individual width/height constraints
    params.mnlw = int(.3/params.n_slots*params.screen_lw)
    params.mxlw = int(.9/params.n_slots*params.screen_lw)
    params.mncw = int(.3*params.screen_cw)
    params.mxcw = int(.9*params.screen_cw)

    if params.direction == 'vertical':
        params.mnh = params.mnlw
        params.mxh = params.mxlw
        params.mnw = params.mncw
        params.mxw = params.mxcw
    if params.direction == 'horizontal':
        params.mnh = params.mncw
        params.mxh = params.mxcw
        params.mnw = params.mnlw
        params.mxw = params.mxlw
    params.mnm = .3/3*params.screen_lw
    params.mxm = .9/3*params.screen_lw

    #margins
    params.m = int(params.screen_lw/10) #m = int(params.screen_lw/7)
    params.tm = 4*params.m
    
    # global suface/height constraints
    params.mnts = 3*params.mnlw*params.mncw
    params.mxts = 3*params.mxlw*params.mxcw
    params.mntl = params.mnlw*3
    if params.direction == 'vertical':params.mxtl = params.screen_height-params.tm #params.mxlw*3
    if params.direction == 'horizontal':params.mxtl = params.screen_width-params.tm
    
    return params

def init_margins(params):
    margins=[]
    #m = int(params.screen_lw/7)
    for i in range(params.n_slots+1):
        margins.append(params.m)
    return margins

def shift_dimensions(slots,params,tl,ts):
    
    # select slot to modify
    i = random.randint(0,params.n_slots-1)
    slot = slots[i]
    # set width, height to add/subtract
    max_range = 10
    h = random.randint(-max_range,max_range)
    w = random.randint(-max_range,max_range)
    
    tl0=tl
    tr = tl+h
    
    # run checks
    checks = [True,True,True,True,True,True,True,True,True]
    if params.direction=="vertical":
        if tl+h<params.mntl:checks[2]=False
        if tl+h>params.mxtl:checks[3]=False
    if params.direction=="horizontal":
        if tl+w<params.mntl:checks[2]=False
        if tl+w>params.mxtl:checks[3]=False
    if slot.height+h<params.mnh:checks[4]=False
    if slot.width+w<params.mnw:checks[5]=False
    if slot.height+h>params.mxh:checks[6]=False
    if slot.width+w>params.mxw:checks[7]=False
    
    if slot.width+w<1*slot.height+h:checks[8]=False
    #if slot.width+w>2*slot.height+h:checks[8]=False
    
    prev_s = slot.width*slot.height
    next_s = (slot.width+w)*(slot.height+h)
    if ts+next_s<params.mnts:checks[0]=False
    if ts+next_s>params.mxts:checks[1]=False
            
    # process the adding/subtraction
    #print(checks)
    if False not in checks:
        slot.height+=h
        slot.width+=w
        ts=ts-prev_s+next_s
        if params.direction=="vertical":tl+=h
        if params.direction=="horizontal":tl+=w    
    return slots,tl,ts




def init_dimensions(params):
    ts=0
    tl=0
    slots=[]
    for i in range(params.n_slots):
        slot = Clay()
        slot.lw = int((params.mnlw+params.mxlw)/2) #int(params.screen_lw/7)
        slot.cw = int((params.mncw+params.mxcw)/2) #int(params.screen_cw/3)
        #slot.cw = slot.lw
        if params.direction=='vertical':
            slot.height = slot.lw = int(slot.lw) #slot.lw #1/2*slot.lw
            slot.width = slot.cw = int(1*slot.cw) #slot.cw #2*slot.cw
        if params.direction=='horizontal':
            slot.height = slot.cw = int(1/1*slot.cw) #slot.cw #1/2*slot.cw
            slot.width = slot.lw = int(slot.lw) #slot.lw #2*slot.lw
        slots.append(slot)
        ts+=slot.cw*slot.lw
        tl+=slot.lw
    return slots,tl,ts




def get_random_type():
    dice = random.random()
    if 0   <= dice <= 1/3: pl=[1,0,0]
    if 1/3 < dice <= 2/3: pl=[0,1,0]
    if 2/3 < dice <= 3/3: pl=[0,0,1]
    return pl
    
def create_sample1(h,w):
    # set main parameters
    params = Clay()
    params.n_slots = 3
    params.screen_height = h #250 #600
    params.screen_width = w #300 #300
    params = set_constraints(params)
    slots,tl,ts = init_dimensions(params)
    margins = init_margins(params)
    slots = set_positions(params,slots,margins)
    
    for i in range(500):
        slots,tl,ts = shift_dimensions(slots,params,tl,ts)
    slots = set_positions(params,slots,margins)

    ##### add semantic criteria
    
    '''
    s0 = slots[0]
    s1 = slots[1]
    if 1.5*s0.width*s0.height < s1.width*s1.height:
        slots[0] = copy.deepcopy(s1)
        slots[1] = copy.deepcopy(s0)
    '''
   
    for i in range(3):slots[i].name = ''

    slots[0].type = get_random_type()
    slots[1].type = get_random_type()
    slots[2].type = get_random_type()
   
    a0 = slots[0].width*slots[0].height
    a1 = slots[1].width*slots[1].height
    a2 = slots[2].width*slots[2].height

    '''
    # if elements are in this order : red,green,blue or green,red,blue
    # the bigger slot between slot0 and slot1 become green and the other one becomes red
    b0 = (slots[0].type == [1,0,0] and slots[1].type == [0,1,0] and slots[2].type == [0,0,1])
    b1 = (slots[1].type == [1,0,0] and slots[0].type == [0,1,0] and slots[2].type == [0,0,1])
    if b0 or b1:
        if a0 > a1:
            slots[0].type = [0,1,0]
            slots[1].type = [1,0,0]
        else:
            slots[0].type = [1,0,0]
            slots[1].type = [0,1,0]

        for i in range(3):slots[i].name+="tag0"

    if not a0 < a1 < a2:
        if slots[0] == [0,0,1] and slots[1] == [0,0,1] and slots[2] == [0,0,1]:
            slots[int(3*random.random())] = [1,0,0]
    '''
    
    '''
    # if slot0 is smaller than slot1 and slot1 is smaller than slot2
    # then each element is blue + each element is stuck on the left border of the banner
    if a0 < a1 < a2:
        slots[0].type = [0,0,1]
        slots[1].type = [0,0,1]
        slots[2].type = [0,0,1]
        for i in range(3):
            slots[i].left = 0
            slots[i].right = slots[i].left+slots[i].width
        for i in range(3):slots[i].name+="tag1 "
    # other wise slots can't all be blue
    if not a0 < a1 < a2:
        if slots[0].type == [0,0,1]:
            if slots[1].type == [0,0,1]:
                if slots[2].type == [0,0,1]:
                    slots[int(3*random.random())].type = [1,0,0]
    '''
    
    #tagRules
    
    # 3 blues => on left border
    if slots[0].type == slots[1].type == slots[2].type == [0,0,1]:
        for i in range(3):
            slots[i].left = 0
            slots[i].right = slots[i].left+slots[i].width

    # anything, green, blue
    '''
    # if last two elements are green and blue : they stick one above the other at the bottom
    if slots[1].type == [0,1,0] and slots[2].type == [0,0,1]:
        slots[2].low = h
        slots[2].top = slots[2].low-slots[2].height
        slots[1].low = slots[2].top-5
        slots[1].top = slots[1].low-slots[1].height
        for i in range(3):slots[i].name+="tag2 "
    '''
    
    # blue green blue => last on bottom, penult few pixels above last
    if slots[0].type == [0,0,1] and slots[1].type == [0,1,0] and slots[2].type == [0,0,1]:
        slots[2].low = h
        slots[2].top = slots[2].low-slots[2].height
        slots[1].low = slots[2].top-5
        slots[1].top = slots[1].low-slots[1].height
        
    # 3 greens => on corners (all expected low left corner)
    if slots[0].type == slots[1].type == slots[2].type == [0,1,0]:
    
        slots[0].width = int(round(.3*w))
        slots[1].width = int(round(.3*w))
        slots[2].width = int(round(.3*w))
        
        slots[0].height = int(round(.3*h))
        slots[1].height = int(round(.3*h))
        slots[2].height = int(round(.3*h))
        
        slots[0].left = 0
        slots[0].top = 0
        
        slots[1].left = w-slots[1].width
        slots[1].top = 0
        
        slots[2].left = w-slots[2].width
        slots[2].top = h-slots[2].height
        
    #RRR
    # 3 reds => all at mid line
    if slots[0].type == slots[1].type == slots[2].type == [1,0,0]:
    
        slots[0].width = int(round(.3*w))
        slots[1].width = int(round(.3*w))
        slots[2].width = int(round(.3*w))
        
        slots[0].height = int(round(.3*w))
        slots[1].height = int(round(.3*w))
        slots[2].height = int(round(.3*w))
        
        slots[0].left = 0
        slots[0].top = int(round(.5*h-.5*int(round(.3*w))))
        
        slots[1].left = int(round(.5*w-.5*int(round(.3*w))))
        slots[1].top = int(round(.5*h-.5*int(round(.3*w))))
        
        slots[2].left = w-int(round(.3*w))
        slots[2].top = int(round(.5*h-.5*int(round(.3*w))))

        for i in range(3):
            slots[i].right = slots[i].left+slots[i].width
            slots[i].low = slots[i].top+slots[i].height
   
    # 2 reds 1 blue => all at top line
    if slots[0].type == slots[1].type == [1,0,0] and slots[2].type == [0,0,1]:
    
        slots[0].width = int(round(.3*w))
        slots[1].width = int(round(.3*w))
        slots[2].width = int(round(.3*w))
        slots[0].height = slots[0].width
        slots[1].height = slots[1].width
        slots[2].height = slots[2].width
        
        slots[0].left = 0
        slots[0].top = 0
        slots[1].left = int(round(.5*w-.5*slots[0].width))
        slots[1].top = 0
        slots[2].left = w-slots[2].width
        slots[2].top = 0

    #BBG
    # 2 blues 1 green => all at bottom line
    if slots[0].type == slots[1].type == [0,0,1] and slots[2].type == [0,1,0]:
    
        slots[0].width = int(round(.3*w))
        slots[1].width = int(round(.3*w))
        slots[2].width = int(round(.3*w))
        slots[0].height = int(round(.3*w))
        slots[1].height = int(round(.3*w))
        slots[2].height = int(round(.3*w))
        
        slots[0].left = 0
        slots[0].top = int(round(h-int(round(.3*w))))
        slots[1].left = int(round(.5*w-.5*int(round(.3*w))))
        slots[1].top = int(round(h-int(round(.3*w))))
        slots[2].left = w-int(round(.3*w))
        slots[2].top = int(round(h-int(round(.3*w))))

    #GGR
    
    # 2 green 1 red => on the right
    if slots[0].type == slots[1].type == [0,1,0] and slots[2].type == [1,0,0]:
        
        slots[0].left = int(round(w-slots[0].width))
        slots[1].left = int(round(w-slots[1].width))
        slots[2].left = int(round(w-slots[2].width))
        
    # 1 blue 1 red 1 blue => diagonal going low right, without overlap
    if slots[0].type == slots[2].type == [0,0,1] and slots[1].type == [1,0,0]:
    
        slots[0].width = int(round(w/3))
        slots[1].width = int(round(w/3))
        slots[2].width = int(round(w/3))
        slots[0].height = int(round(h/3))
        slots[1].height = int(round(h/3))
        slots[2].height = int(round(h/3))
        
        slots[0].left = 0
        slots[0].top = 0
        slots[1].left = slots[0].left+int(round(w/3))
        slots[1].top = slots[0].top+int(round(h/3))
        
        slots[2].left = slots[1].left+int(round(w/3))
        slots[2].top = slots[1].top+int(round(h/3))
        
    # 1 red 1 blue 1 red => diagonal going low left, without overlap
    if slots[0].type == slots[2].type == [1,0,0] and slots[1].type == [0,0,1]:
    
        slots[0].width = int(round(w/3))
        slots[1].width = int(round(w/3))
        slots[2].width = int(round(w/3))
        slots[0].height = int(round(h/3))
        slots[1].height = int(round(h/3))
        slots[2].height = int(round(h/3))
        
        slots[0].left = w-slots[0].width
        slots[0].top = 0
        slots[1].left = slots[0].left-slots[1].width
        slots[1].top = slots[0].top+slots[0].height
        slots[2].left = slots[1].left-slots[2].width
        slots[2].top = slots[1].top+slots[1].height
        
    #RGB 
    # 1 red 1 green 1 blue => diagonal with overlap, centered on first (red) elem
    if slots[0].type == [1,0,0] and slots[1].type == [0,1,0] and slots[2].type == [0,0,1]:
        
        slots[0].width = int(round(w/3))
        slots[1].width = int(round(w/3))
        slots[2].width = int(round(w/3))
        slots[0].height = slots[0].width
        slots[1].height = slots[1].width
        slots[2].height = slots[2].width
    
        slots[1].left = int(round(w/2-slots[1].width/2))
        slots[1].top = int(round(h/2-slots[1].height/2))
        slots[0].left = slots[1].left-int(round(slots[0].width/2))
        slots[0].top = slots[1].top-int(round(slots[0].height/2))
        slots[2].left = slots[1].left+int(round(slots[2].width/2))
        slots[2].top = slots[1].top+int(round(slots[2].height/2))
        
    for i in range(3):
        slots[i].right = slots[i].left+slots[i].width
        slots[i].low = slots[i].top+slots[i].height
        
    return slots,params

# BBB  => on left border
# BGB  => last on bottom, penultimate few pixels above last
# GGG  => on corners (all expected low left corner)
# RRR  => all at mid line
# RRB  => all at top line
# BBG  => all at bottom line
# GGR  => on the right
# BRB  => diagonal going low right, without overlap
# RBR  => diagonal going low left, without overlap
# RGB  => diagonal with overlap, centered on first (red) elem




def create_synth1(n_samples):

    #################################### CREATE SYNTH SAMPLES

    #n_samples = 100000#100000#500000#10000
    print("start",time.ctime())
    samples=[]
    #w,h = 728,90 #728,90 #300,600 #300,250

    for num_sample in range(n_samples):        
        #'''
        h = 600
        w = 300
        #'''
        if len(samples)%10000==0:print(len(samples),time.ctime())
        slots,params = create_sample1(h,w)

        '''
        for s in slots:
            #if s.left+s.w>300:print("tag0 : error",s.left+s.w)
            if s.right>300:print("tag0 : error",s.right)
        '''

        samples.append([slots,h,w])
    print(len(samples),time.ctime())
    

    #################################### ARCHIVE CREATED RAW SYNTH SAMPLES

    n_slots=3
    npa = np.zeros((len(samples),2),dtype=int)
    npa1 = np.zeros((len(samples),n_slots,4),dtype=int)
    npa2 = np.zeros((len(samples),n_slots,3),dtype=int)
    for num_sample in range(len(samples)):
        for num_slot in range(n_slots):
            npa[num_sample,0] = samples[num_sample][1] # height
            npa[num_sample,1] = samples[num_sample][2] # width
            
            npa1[num_sample,num_slot,0]=samples[num_sample][0][num_slot].top
            npa1[num_sample,num_slot,1]=samples[num_sample][0][num_slot].left
            npa1[num_sample,num_slot,2]=samples[num_sample][0][num_slot].low
            npa1[num_sample,num_slot,3]=samples[num_sample][0][num_slot].right
            
            npa2[num_sample,num_slot,0]=samples[num_sample][0][num_slot].type[0]
            npa2[num_sample,num_slot,1]=samples[num_sample][0][num_slot].type[1]
            npa2[num_sample,num_slot,2]=samples[num_sample][0][num_slot].type[2]

    '''
    root_path = '/home/paintedpalms/rdrive/taff/data/automated_layout/expS10'
    np.save(root_path+'/synth_samples_semantics.npy',npa2)
    np.save(root_path+'/synth_samples_coordinates.npy',npa1)
    np.save(root_path+'/synth_samples_screens_dimensions.npy',npa)
    '''

    #################################### SEPARATED ARCHIVES => RAW

    '''
    screens_dimensions = np.load(root_path+'/synth_samples_screens_dimensions.npy')
    coordinates = np.load(root_path+'/synth_samples_coordinates.npy')
    semantics = np.load(root_path+'/synth_samples_semantics.npy')
    '''

    screens_dimensions = npa
    coordinates = npa1
    semantics = npa2

    n_samples,n_slots,dummy = np.shape(coordinates)

    # GET RAW X + RAW Y

    x_raw = np.zeros((n_samples,17),dtype=int)
    y_raw = np.zeros((n_samples,6),dtype=int)
    deltas_raw = np.zeros((n_samples,n_slots),dtype=np.float)

    for i in range(n_samples):
        
        # deltas
        delta0 = 0.5+random.random()
        delta1 = 0.5+random.random()
        delta2 = 0.5+random.random()
        
        # screen dimensions
        h = screens_dimensions[i,0]
        w = screens_dimensions[i,1]
        
        # slots dimensions
        h0 = (coordinates[i,0,2]-coordinates[i,0,0]) # slot0 original h
        h1 = (coordinates[i,1,2]-coordinates[i,1,0]) # slot1 original h
        h2 = (coordinates[i,2,2]-coordinates[i,2,0]) # slot2 original h
        w0 = (coordinates[i,0,3]-coordinates[i,0,1]) # slot0 original w
        w1 = (coordinates[i,1,3]-coordinates[i,1,1]) # slot1 original w
        w2 = (coordinates[i,2,3]-coordinates[i,2,1]) # slot2 original w
        
        # slots positions
        top0 = coordinates[i,0,0]
        top1 = coordinates[i,1,0]
        top2 = coordinates[i,2,0]
        left0 = coordinates[i,0,1]
        left1 = coordinates[i,1,1]
        left2 = coordinates[i,2,1]

        #tag1

        # slots categories
        type0 = semantics[i,0]
        type1 = semantics[i,1]
        type2 = semantics[i,2]
        
        # deltas
        deltas_raw[i,0] = delta0
        deltas_raw[i,1] = delta1
        deltas_raw[i,2] = delta2
        
        # screen
        x_raw[i,0] = h
        x_raw[i,1] = w
        
        # slot0
        x_raw[i,2] = int(round(h0*delta0)) # slot0 twisted h
        x_raw[i,3] = int(round(w0*delta0)) # slot0 twisted w
        x_raw[i,4] = type0[0] # slot0 semantic 0
        x_raw[i,5] = type0[1] # slot0 semantic 1
        x_raw[i,6] = type0[2] # slot0 semantic 2
        
        # slot1
        x_raw[i,7] = int(round(h1*delta1)) # slot1 twisted h
        x_raw[i,8] = int(round(w1*delta1)) # slot1 twisted w
        x_raw[i,9] = type1[0] # slot1 semantic 0
        x_raw[i,10] = type1[1] # slot1 semantic 1
        x_raw[i,11] = type1[2] # slot1 semantic 2

        # slot2
        x_raw[i,12] = int(round(h2*delta2)) # slot2 twisted h
        x_raw[i,13] = int(round(w2*delta2)) # slot2 twisted w
        x_raw[i,14] = type2[0] # slot2 semantic 0
        x_raw[i,15] = type2[1] # slot2 semantic 1
        x_raw[i,16] = type2[2] # slot2 semantic 2

        # slot0
        y_raw[i,0] = top0 # slot0 top
        y_raw[i,1] = left0 # slot0 left
        deltas_raw[i,0] = delta0 # slot0 delta

        # slot1
        y_raw[i,2] = top1 # slot1 top
        y_raw[i,3] = left1 # slot1 left
        deltas_raw[i,1] = delta1 # slot1 delta

        # slot2
        y_raw[i,4] = top2 # slot2 top
        y_raw[i,5] = left2 # slot2 left
        deltas_raw[i,2] = delta2 # slot2 delta
        
    '''
    np.save(root_path+'/x_raw.npy',x_raw)
    np.save(root_path+'/y_raw.npy',y_raw)
    np.save(root_path+'/deltas_raw.npy',deltas_raw)
    '''

    x_norm = np.zeros(np.shape(x_raw),dtype=float)
    y_norm = np.zeros((len(y_raw),9),dtype=float)

    n_samples = len(y_norm)
    n_slots = 3

    for i in range(n_samples):
        
        # screen dimensions
        x_norm[i,0] = x_raw[i,0]/1000
        x_norm[i,1] = x_raw[i,1]/1000
        
        # slot0
        x_norm[i,2] = x_raw[i,2]/1000 # slot0 twisted h
        x_norm[i,3] = x_raw[i,3]/1000 # slot0 twisted w
        x_norm[i,4] = x_raw[i,4]      # slot0 category score 0
        x_norm[i,5] = x_raw[i,5]      # slot0 category score 1
        x_norm[i,6] = x_raw[i,6]      # slot0 category score 2

        # slot1
        x_norm[i,7] = x_raw[i,7]/1000
        x_norm[i,8] = x_raw[i,8]/1000
        x_norm[i,9] = x_raw[i,9]
        x_norm[i,10] = x_raw[i,10]
        x_norm[i,11] = x_raw[i,11]

        # slot2
        x_norm[i,12] = x_raw[i,12]/1000
        x_norm[i,13] = x_raw[i,13]/1000
        x_norm[i,14] = x_raw[i,14]
        x_norm[i,15] = x_raw[i,15]
        x_norm[i,16] = x_raw[i,16]
        
        # slot0
        y_norm[i,0] = y_raw[i,0]/1000 # slot0 top
        y_norm[i,1] = y_raw[i,1]/1000 # slot0 left
        y_norm[i,2] = deltas_raw[i,0]/1.5 # slot0 delta

        # slot1
        y_norm[i,3] = y_raw[i,2]/1000
        y_norm[i,4] = y_raw[i,3]/1000
        y_norm[i,5] = deltas_raw[i,1]/1.5

        # slot2
        y_norm[i,6] = y_raw[i,4]/1000
        y_norm[i,7] = y_raw[i,5]/1000
        y_norm[i,8] = deltas_raw[i,2]/1.5
        
    np.save('x_synth1.npy',x_norm)
    np.save('y_synth1.npy',y_norm)


def display_synth_samples_1(x_norm,y_norm,n,rule):

    #n_samples = len(x_norm)
    n_samples=1000*n
    n_slots = 3

    x_raw = np.zeros(np.shape(x_norm),dtype=int)
    y_raw = np.zeros((n_samples,6),dtype=int)
    deltas_raw = np.zeros((n_samples,n_slots),dtype=float)

    for i in range(n_samples):
        
        # screen dimensions
        x_raw[i,0] = int(np.round(x_norm[i,0]*1000))
        x_raw[i,1] = int(np.round(x_norm[i,1]*1000))
        
        # slot0
        x_raw[i,2] = int(np.round(x_norm[i,2]*1000)) # slot0 twisted h
        x_raw[i,3] = int(np.round(x_norm[i,3]*1000)) # slot0 twisted w
        x_raw[i,4] = int(np.round(x_norm[i,4]))      # slot0 category score 0
        x_raw[i,5] = int(np.round(x_norm[i,5]))      # slot0 category score 1
        x_raw[i,6] = int(np.round(x_norm[i,6]))      # slot0 category score 2

        # slot1
        x_raw[i,7] = int(np.round(x_norm[i,7]*1000))
        x_raw[i,8] = int(np.round(x_norm[i,8]*1000))
        x_raw[i,9] = int(np.round(x_norm[i,9]))
        x_raw[i,10] = int(np.round(x_norm[i,10]))
        x_raw[i,11] = int(np.round(x_norm[i,11]))

        # slot2
        x_raw[i,12] = int(np.round(x_norm[i,12]*1000))
        x_raw[i,13] = int(np.round(x_norm[i,13]*1000))
        x_raw[i,14] = int(np.round(x_norm[i,14]))
        x_raw[i,15] = int(np.round(x_norm[i,15]))
        x_raw[i,16] = int(np.round(x_norm[i,16]))

        # slot0
        y_raw[i,0] = int(np.round(y_norm[i,0]*1000)) # slot0 top
        y_raw[i,1] = int(np.round(y_norm[i,1]*1000)) # slot0 left
        deltas_raw[i,0] = y_norm[i,2]*1.5 # slot0 delta

        # slot1
        y_raw[i,2] = int(np.round(y_norm[i,3]*1000))
        y_raw[i,3] = int(np.round(y_norm[i,4]*1000))
        deltas_raw[i,1] = y_norm[i,5]*1.5

        # slot2
        y_raw[i,4] = int(np.round(y_norm[i,6]*1000))
        y_raw[i,5] = int(np.round(y_norm[i,7]*1000))
        deltas_raw[i,2] = y_norm[i,8]*1.5

    n_slots = 3
    new_samples = []
    for num_sample in range(n_samples):

        kx = 2
        ky = 0
        slots = []
        for num_slot in range(n_slots):

            slot = Clay()
            slot.top = int(y_raw[num_sample,ky+0])
            slot.left = int(y_raw[num_sample,ky+1])

            slot.h =  int(x_raw[num_sample,kx+0]/deltas_raw[num_sample,num_slot])
            slot.w =  int(x_raw[num_sample,kx+1]/deltas_raw[num_sample,num_slot])
            slot.low = int(slot.top+slot.h)
            slot.right = int(slot.left+slot.w)

            if slot.right > 300: print("tag2 : error", slot.right)

            type0 = int(x_raw[num_sample,kx+2])
            type1 = int(x_raw[num_sample,kx+3])
            type2 = int(x_raw[num_sample,kx+4])
            slot.type = np.asarray([type0,type1,type2])

            slot.type = list(slot.type)
            slots.append(slot)

            kx+=5
            ky+=2

        new_samples.append(slots)

    k=0
    for sample in new_samples:
        slots = sample#[0]

        '''
        s0=slots[0]
        s1=slots[1]
        s2=slots[2]
        print("right 0",s0.left+s0.w)
        print("right 1",s1.left+s1.w)
        print("right 2",s2.left+s2.w)
        '''

        if k<n:
            if sample_is_concerned_by_rule(slots,rule):
                show_screen5_ancien(slots,600,300)
                k+=1

# get type of a synth1 sample
def get_types_trigram(slots):
    tp=""
    for s in slots:
        if s.type == [1, 0, 0]: tp += "r"
        if s.type == [0, 1, 0]: tp += "g"
        if s.type == [0, 0, 1]: tp += "b"
    return tp

def sample_is_concerned_by_rule(slots,rule):
    tp=get_types_trigram(slots)
    spec_combinations=get_special_combinations_trigram()
    ok = False
    if rule=="general" and tp not in spec_combinations:ok=True
    if rule==tp:ok=True
    return ok

def add_shape_color5_ancien(npa,top,left,low,right,r,g,b):    
    for i in range(len(npa)):
        for j in range(len(npa[0])):
            if top<=i<=low and left<=j<=right:
                npa[i,j,0] = r
                npa[i,j,1] = g
                npa[i,j,2] = b
    return npa

def show_screen5_ancien(slots,screen_h,screen_w):
    npa = init_screen_ancien(screen_h,screen_w)
    for num_slot in range(len(slots)):
        slot = slots[num_slot]
        
        '''
        if num_slot == 0:(r,g,b) = (255,100,100)
        if num_slot == 1:(r,g,b) = (100,255,100)
        if num_slot == 2:(r,g,b) = (100,100,255)
        '''
        
        [r,g,b] = 155*np.asarray(slot.type)+100
        
        npa = add_shape_color5_ancien(npa,slot.top,slot.left,slot.low,slot.right,r,g,b)
    npa=resize_image(npa, 100)
    display(get_image_from_npa(npa))

def init_screen_ancien(h,w):    
    npa = np.zeros((h,w,4),dtype=np.uint8)
    for i in range(len(npa)):
        for j in range(len(npa[0])):
            v=100
            for k in [i,j]:
                if k%100==0:v=70
            npa[i,j,0] = v
            npa[i,j,1] = v
            npa[i,j,2] = v
            npa[i,j,3] = 255
    return npa
            