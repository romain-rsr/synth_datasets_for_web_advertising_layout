

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        public imports
#------------------------------------------------------------------------------------------------------------------------------------

# ok

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        private imports
#------------------------------------------------------------------------------------------------------------------------------------

import sys
from importlib import reload

p_code="/home/paintedpalms/rdrive/taff/code"
sys.path.insert(0,p_code)

from basics import *

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        quant eval : diversity
#------------------------------------------------------------------------------------------------------------------------------------

# measure diversity in samples generated by layout gan
def diversity_eval(samples):
    combinaisons=[]
    for sample in samples:
        combinaison=[]
        for asset in sample.assets:
            combinaison.append(asset.type)
        if combinaison not in combinaisons:combinaisons.append(combinaison)
    return len(combinaisons)#/max(1,len(samples))

#------------------------------------------------------------------------------------------------------------------------------------
#                                                        quant eval : error
#------------------------------------------------------------------------------------------------------------------------------------

# eval samples concerned by general rules only
def quant_eval(samples,str_option):

    h=600
    w=300
        
    top_overtakings=[]
    left_overtakings=[]
    right_overtakings=[]
    low_overtakings=[]
    overtakings=[]
    overlaps_btw_assets=[]
    
    n_samples=len(samples)
    n_assets=3
    n_samples_in_error=0
    n_assets_in_error=0
    
    # overlap or exceeding screen limits => error
    for sample in samples:
        prev_low=0
        sample_is_in_error=0
        for i_asset in range(n_assets):
            asset_is_in_error=0
            asset=sample.assets[i_asset]
            if i_asset==0 and asset.top<0:
                asset_is_in_error=1
                sample_is_in_error=1
                top_overtakings.append(0-asset.top)
            if i_asset==n_assets-1 and asset.low>h:
                asset_is_in_error=1
                sample_is_in_error=1
                low_overtakings.append(asset.low-h)
            if asset.left<0:
                asset_is_in_error=1
                sample_is_in_error=1
                left_overtakings.append(0-asset.left)
            if asset.right>w:
                asset_is_in_error=1
                sample_is_in_error=1
                right_overtakings.append(asset.right-w)
            if i_asset>0 and asset.top<prev_low:
                asset_is_in_error=1
                sample_is_in_error=1              
                overlaps_btw_assets.append(prev_low-asset.top)
                
            prev_low=asset.low
                
            n_assets_in_error+=asset_is_in_error
        n_samples_in_error+=sample_is_in_error
        
    for values in left_overtakings,top_overtakings,right_overtakings,low_overtakings:
        for v in values:overtakings.append(v)
        
    results={}
    results["overtakings"]={}
    results["overtakings"]["all"]=overtakings
    results["overtakings"]["left"]=left_overtakings
    results["overtakings"]["top"]=top_overtakings
    results["overtakings"]["right"]=right_overtakings
    results["overtakings"]["low"]=low_overtakings
    results["overlaps"]=overlaps_btw_assets
    results["n_samples"]=n_samples
    results["n_samples_in_error"]=n_samples_in_error
    results["n_assets"]=n_assets*n_samples
    results["n_assets_in_error"]=n_assets_in_error
    results["samples_in_error"]=n_samples_in_error/max(1,n_samples)
    results["assets_in_error"]=n_assets_in_error/max(1,(n_samples*n_assets))

    if 0==1:

        name="samples in error"
        
        if len(name)<=7:after_name="\t\t\t:"
        if len(name)>7:after_name="\t\t:"
        if len(name)>15:after_name="\t:"
    
    if 1==1:
        
        s=""
        
        s+="\n"
        s+="quant eval"+"\n"
        s+="\n"

   
        for k0 in results.keys():
 
            if type(results[k0])!=dict:
                if str_option==0:display_stats_line(k0,results[k0])
                if str_option==1:
                    if k0 in ["samples_in_error","assets_in_error"]:
                        s+=get_str_with_tabs(k0,results[k0],3)#+"\n"
                  
        for k0 in results.keys():
            if type(results[k0])==dict:
                for k1 in results[k0].keys():
                    if str_option==0:display_stats_line(k1,results[k0][k1])
                    if str_option==1:
                        if k1 in ["samples_in_error","assets_in_error"]:
                            s+=get_str_with_tabs(k1,results[k0][k1],3)#+"\n"
                            

    return s,n_samples_in_error/max(1,n_samples)
  
#------------------------------------------------------------------------------------------------------------------------------------
#                                                        quant eval : error
#------------------------------------------------------------------------------------------------------------------------------------

def quant_eval0(samples):
    m=0
    w=300
    h=600
    k=0
    k_bench=0
    ks=0
    s=0
    k_overlap=0
    for sample in samples:
        k_bench+=1
        i=0
        prev_low=0
        for asset in sample.assets:
            if i>0:prev_low=sample.assets[i-1].low
            next_top=h
            if i<2:next_top=sample.assets[i+1].top
            i+=1
            k_temp=k
            if asset.top<prev_low-m*2:
                k+=1
                s+=abs(asset.top-prev_low)
                k_overlap+=1
            '''
            if asset.low>next_top+m*2:
                k+=1
                s+=abs(asset.low)
                k_overlap+=1
            '''
            if asset.right>w+m:
                k+=1
                s+=abs(asset.right-w)
            if asset.low>h+m*2:
                k+=1
                s+=abs(asset.low-h)
            if asset.left<0-m:
                k+=1
                s+=abs(asset.left)
            '''
            if asset.top<0-m*2:
                k+=1
                s+=abs(asset.top)
            '''
            if k>k_temp:ks+=1
    if ks==0:print("eval",k_bench,k,s)
    if ks!=0:print("eval",k_bench,k,s/ks)

               
#features distributions (no error measurement !!!)

def init_stats(c):
    stats={}
    stats["lefts"]=[]
    stats["tops"]=[]
    stats["rights"]=[]
    stats["last lows"]=[]
    stats["lateral mids"]=[]
    stats["vertical mids"]=[]
    
    for i_asset in range(c.n_assets):
        stats[i_asset]={}
        for tp in c.types:
            stats[i_asset][tp]={}
            for ft in c.features:
                stats[i_asset][tp][ft]=[]
    return stats

def get_stats(c,samples):
    stats=init_stats(c)
    for i_sample in range(len(samples)):
        sample=samples[i_sample]
        for i_asset in range(c.n_assets):
            asset=sample.assets[i_asset]
            
            # main stats
            stats["lefts"].append((asset.left))
            stats["tops"].append((asset.top))
            stats["lateral mids"].append(int(np.round(asset.left+(asset.right-asset.left)/2)))
            stats["vertical mids"].append(int(np.round(asset.top+(asset.low-asset.top)/2)))
            stats["rights"].append(asset.right)
            if i_asset==c.n_assets-1:
                stats["last lows"].append(asset.low)
                
            # all stats
            for i_feature in range(c.n_features):
                
                #print("okok",i_sample,i_asset,asset.type)
                
                '''
                print("---------------------------------------------")
                print("i_sample,i_asset",i_sample,i_asset)
                print("width",sample.assets[i_asset].width)
                print(asset.type)
                print(stats[i_asset])
                print(stats[i_asset][asset.type])
                print(stats[i_asset][asset.type]["width"])
                print(sample.assets[i_asset].width)
                '''
                
                stats[i_asset][asset.type]["width"].append(sample.assets[i_asset].width)
                stats[i_asset][asset.type]["height"].append(sample.assets[i_asset].height)
                stats[i_asset][asset.type]["left"].append(sample.assets[i_asset].left)
                stats[i_asset][asset.type]["top"].append(sample.assets[i_asset].top)
                stats[i_asset][asset.type]["right"].append(sample.assets[i_asset].right)
                stats[i_asset][asset.type]["low"].append(sample.assets[i_asset].low)
                
    return stats

#tagtag

def save_stats_str(c,stats):

    s=""
    s+="------------------ main stats"+"\n"
    for name in "lateral mids","vertical mids","lefts","rights","tops","last lows":
        values=stats[name]
        if name in ["rights","lefts","tops"]:name+="\t"
        s+=name+"\t:"+"\t"+str(min(values))+"\t"+str(max(values))+"\t"+str(int(np.round(np.mean(values))))+"\t"+str(int(np.round(np.std(values))))+"\n"
    s+="\n"

    s+="------------------ all stats"+"\n"
    for i_asset in range(c.n_assets):
        for tp in c.types:
            for ft in c.features:
                values=stats[i_asset][tp][ft]
                if len(values)==0:
                    s+=str(i_asset)+" "+str(tp)+" "+str(ft)+"\t:"+"\t no values"+"\n"
                if len(values)> 0:
                    s+=str(i_asset)+" "+str(tp)+" "+str(ft)+"\t:"+"\t"+str(min(values))+"\t"+str(max(values))+"\t"+str(int(np.round(np.mean(values))))+"\t"+str(int(np.round(np.std(values))))+"\n"
    s+="\n"
    
    save_text(c.results_folder+"/stats.txt",s)
    
def save_text(p,s):
    file = open(p,"w")
    file.write(s)
    file.close()

def display_stats(c,stats):
    print("------------------ main stats")
    for name in "lateral mids","vertical mids","lefts","rights","tops","last lows":
        values=stats[name]
        if name in ["rights","lefts","tops"]:name+="\t"
        print(name,"\t:","\t",min(values),"\t",max(values),"\t",int(np.round(np.mean(values))),"\t",int(np.round(np.std(values))))
    print("")

    print("------------------ all stats")
    for i_asset in range(c.n_assets):
        for tp in c.types:        
            for ft in c.features:
                values=stats[i_asset][tp][ft]
                if len(values)==0:print(i_asset,tp,ft,"\t:","\t no values")
                if len(values)> 0:
                    print(i_asset,tp,ft,"\t:","\t",min(values),"\t",max(values),"\t",int(np.round(np.mean(values))),"\t",int(np.round(np.std(values))))
    print("")
    
    
    
    
'''
############################################## extract data distributions

def extract_distributions(samples):
    w=300
    h=600
    n_assets=len(samples[0].assets)
    params=[]
    for asset_type in "text","image","cta","logo":
        p=Clay()
        n=0
        lefts,tops,rights,lows=[],[],[],[]
        widths=[]
        heights=[]
        ranks=[]
        abs_ranks=[]
        prev_spaces=[]
        next_spaces=[]
        global_heights=[]
        areas=[]
        width_height_ratios=[]
        for sample in samples:
            global_height=0
            for i_asset in range(n_assets):
                asset=sample.assets[i_asset]
                global_height+=asset.height
                if asset.type==asset_type:
                    n+=1
                    ranks.append(i_asset)
                    abs_ranks.append(abs(1-i_asset))
                    lefts.append(asset.left)
                    rights.append(asset.right)
                    tops.append(asset.top)
                    lows.append(asset.low)
                    widths.append(asset.width)
                    heights.append(asset.height)
                    width_height_ratios.append(asset.height/asset.width)
                    areas.append(asset.width*asset.height)
                    if i_asset==0:prev_spaces.append(asset.top)
                    if i_asset==n_assets-1:next_spaces.append(h-asset.low)
            global_heights.append(global_height)
        zeros=0
        ones=0
        twos=0
        for rank in ranks:
            if rank==0:zeros+=1
            if rank==1:ones+=1
            if rank==2:twos+=1
        c.asset_type=asset_type
        c.ranks=[zeros,ones,twos]
        c.left=np.average(lefts),np.std(lefts)
        c.top=np.average(tops),np.std(tops)
        c.right=np.average(rights),np.std(rights)
        c.low=np.average(lows),np.std(lows)

        c.w=np.average(widths),np.std(widths)
        c.h=np.average(heights),np.std(heights)
        c.hw=np.average(width_height_ratios),np.std(width_height_ratios)
        c.th=np.average(global_heights),np.std(global_heights)
        c.ps=np.average(prev_spaces),np.std(prev_spaces)
        c.ns=np.average(widths),np.std(widths)
        params.append(p)
    return params

    
def display_distributions(params):
    for p in params:
        if sum(p.ranks)>0:
            print("------------------",p.asset_type)
            names=["ranks","w","h","th","left","right","top","low","ps","ns",]
            for name in names:#[1:]:
                print(name,getattr(p, name))

''' 
print(end="")