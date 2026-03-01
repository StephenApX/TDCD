import matplotlib.pyplot as plt
from skimage.io import imread
from torchange.models.segment_any_change import AnyChange
from torchange.models.segment_any_change.segment_anything.utils.amg import (
    rle_to_mask
)
import numpy as np
import cv2
import os
from PIL import Image
from skimage.filters.thresholding import threshold_multiotsu

def cal_cos_smilarity_float(prev_inte_hire_features, curr_inte_hire_features):
    multi_dotsum_prev_curr = np.sum(prev_inte_hire_features * curr_inte_hire_features, axis=2)[:,:,np.newaxis]
    
    prev_norm = np.linalg.norm(prev_inte_hire_features, axis=2, keepdims=True)
    curr_norm = np.linalg.norm(curr_inte_hire_features, axis=2, keepdims=True)
    multi_dis = prev_norm * curr_norm

    cos_float01 = (multi_dotsum_prev_curr / multi_dis) * 0.5 + 0.5
    return cos_float01

def otsu_thres(delta):
    val=np.zeros([256])
    for th in range(256):
        loc1=delta>th
        loc2=delta<=th
        '''
            delta[loc1]=255
            delta[loc2]=0
        '''
        if delta[loc1].size==0:
            mu1=0
            omega1=0
        else:
            mu1=np.mean(delta[loc1])
            omega1=delta[loc1].size/delta.size

        if delta[loc2].size==0:
            mu2=0
            omega2=0
        else:
            mu2=np.mean(delta[loc2])
            omega2=delta[loc2].size/delta.size
        val[th]=omega1*omega2*np.power((mu1-mu2),2)
    loc = np.where(val==np.max(val))
    return loc[0]



dataname = 'levir1024' # second200 levir1024  sysu800 cnam200
prev_img_dir = "dataset\\ChangeDetection\\LEVIR-CD\\test\\A\\"
curr_img_dir = "dataset\\ChangeDetection\\LEVIR-CD\\test\\B\\"

img_names = []
for f in os.listdir(prev_img_dir):
    if str(f).endswith('.png'):
        img_names.append(f)


exp_dir = f'exp\\{dataname}\\11_sam_feat_cos' #    exte_SemiCD_Inst_SAM

os.makedirs(exp_dir, exist_ok=True)
os.makedirs(os.path.join(exp_dir, 'cosdis'), exist_ok=True)
os.makedirs(os.path.join(exp_dir, '255'), exist_ok=True)

# initialize AnyChange  
# customize the hyperparameters of SAM's mask generator 
m.make_mask_generator(
    points_per_side=32,
    stability_score_thresh=0.95,
)
# customize your AnyChange's hyperparameters
m.set_hyperparameters(
    change_confidence_threshold=145,
    use_normalized_feature=True,
    bitemporal_match=True,
)




def weighted_prob_with_sam_masks(masks, prob_map):

    weighted_prob = np.zeros_like(prob_map)
    for idx in range(len(masks["rles"])):
        mask_01 = rle_to_mask(masks["rles"][idx]).astype(np.uint8).astype(np.float32) # 0 1
        # mask_01 = np.where(slic_map==n,1,0).astype(np.uint8).astype(np.float32)
        masked_prob = np.multiply(prob_map, np.expand_dims(mask_01, -1))
        sum_of_each_channel = np.array([np.sum(masked_prob[:,:,c]) for c in range(masked_prob.shape[-1])], dtype=np.float32) 
        ave_of_each_channel = sum_of_each_channel / np.sum(mask_01==1)
        masked_ave_prob = np.multiply(np.expand_dims(mask_01, -1), ave_of_each_channel) 
        weighted_prob += masked_ave_prob
    
    non_weight_mask = np.where(weighted_prob>0, 0, 1).astype(np.uint8).astype(np.float32) # 0 1
    non_weight_prob = np.multiply(prob_map, non_weight_mask)
    
    out_prob = weighted_prob + non_weight_prob
    return out_prob


def compare_embedding(t1_embed, t2_embed, h, w):
    t1_embed = t1_embed.cpu().numpy().squeeze().transpose([1,2,0]) # out: channel last
    t2_embed = t2_embed.cpu().numpy().squeeze().transpose([1,2,0])

    t1_embed_resize = cv2.resize(t1_embed, (h, w), interpolation=cv2.INTER_LINEAR)
    t2_embed_resize = cv2.resize(t2_embed, (h, w), interpolation=cv2.INTER_LINEAR)

    dm_cons_cossim = cal_cos_smilarity_float(t1_embed_resize, t2_embed_resize)

    return dm_cons_cossim


conc_df = None
res_dict = {}
for i, img_name in enumerate(img_names):
    print(img_name)

    prev_img_path = os.path.join(prev_img_dir, img_name)
    curr_img_path = os.path.join(curr_img_dir, img_name)
    img1 = imread(prev_img_path)
    img2 = imread(curr_img_path)
    width, height = np.array(img1).shape[:2]

    # SAM pred
    _, return_data = m.forward(img1, img2) 
    t1_mask_data, t1_image_embedding = return_data['t1_mask_data'], return_data['t1_image_embedding']
    t2_mask_data, t2_image_embedding = return_data['t2_mask_data'], return_data['t2_image_embedding']

    
    # SAM feat Cd.
    feat_cos_sim = compare_embedding(t1_image_embedding, t2_image_embedding, width, height)

    # save dis
    cons_cosdis = 1 - feat_cos_sim
    cons_cosdis = np.clip(cons_cosdis, 0, 1).astype(np.float32).squeeze()
    
    cos_dis_uint8 = (cons_cosdis * 255).astype(np.uint8)
    # collect non-zero results.
    if i == 0:
        conc_df = cos_dis_uint8[cos_dis_uint8 > 0]
    else:
        conc_df = np.concatenate([conc_df, cos_dis_uint8[cos_dis_uint8>0]], axis=0)
    '''
    Save info.
    '''
    res_dict[img_name] = {}
    res_dict[img_name]['dis'] = cos_dis_uint8

    
    '''
    Save cos-dis map.        
    '''
    out_cos_path = os.path.join(exp_dir, 'cosdis', img_name)
    cv2.imwrite(out_cos_path, cos_dis_uint8)
    m.clear_cached_embedding()




'''
Global OTSU.
'''
# thres = otsu_thres(conc_df)
m2_threshold = threshold_multiotsu(conc_df, classes=2)
m3_thresholds = threshold_multiotsu(conc_df)
print(m2_threshold, m3_thresholds)
'''
Save BCD map.
'''
for img_name, img_dict in res_dict.items():
    cos_dis_uint8 = img_dict['dis']
    df_int8_0 = np.where(cos_dis_uint8>=m2_threshold, 1, 0).astype(np.uint8)
    
    df_int8 = np.where(cos_dis_uint8>=m2_threshold, 255, 0).astype(np.uint8)
    out_png_path = os.path.join(exp_dir, '255', img_name)
    cv2.imwrite(out_png_path, df_int8)