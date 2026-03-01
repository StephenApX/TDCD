from PIL import Image
import numpy as np
import cv2
import os 
from skimage.segmentation import slic, mark_boundaries


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


img_dir = "dataset\\ChangeDetection\\LEVIR-CD\\test\\A_B"
img_name_list = []
for f in os.listdir(img_dir):
    if str(f).endswith('.png'):
        img_name_list.append(f)




out_dir = 'repo\\sscm25_exp\\exp\\levir1024\\01_sclip_l2norm_gotsu' #   
os.makedirs(os.path.join(out_dir, 'cosdis'), exist_ok=True)
os.makedirs(os.path.join(out_dir, '255'), exist_ok=True)

config_file = "cfg_seg_levir18.py" 


from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmengine.config import Config
from mmengine.runner import Runner


cfg = Config.fromfile(config_file)
cfg.launcher = 'none'
cfg.work_dir = out_dir
print(cfg)

runner = Runner.from_cfg(cfg)
runner.load_or_resume()

print(runner.model)
model = runner.model
model.cfg = cfg  # save the config in the model for convenience
model.to('cuda:0')
model.eval()

def infer(img_file, model):
    
    image = Image.open(img_file)
    width, height = np.array(image).shape[:2]
    results = inference_model(model, img_file)
    logits = np.array(results.seg_logits.data.cpu().numpy())

    logits = logits.transpose([1,2,0]).astype(np.float32)
    prob_map = np.zeros((width, height, 2)) # c
    bld_logits = logits[:,:,:8]
    nonbld_logits = logits[:,:,8:]

    sum_bld_prob = np.sum(bld_logits, axis=2)
    prob_map[:,:,0] = sum_bld_prob
    sum_nonbld_prob = np.sum(nonbld_logits, axis=2)
    prob_map[:,:,1] = sum_nonbld_prob
    return prob_map


# divide img list.
pair_img_list = []
for img_name in img_name_list:
    if str(img_name).startswith('A_'):
        pair_img_list.append([img_name, str(img_name).replace('A_', 'B_')])


conc_df = None
res_dict = {}
for i, img_pair in enumerate(pair_img_list):
    prev_img_name, curr_img_name = img_pair[:]
    print(prev_img_name, curr_img_name)
    cd_cosdis_name = str(prev_img_name).replace('A_', '')


    prev_img_file = os.path.join(img_dir, prev_img_name)
    curr_img_file = os.path.join(img_dir, curr_img_name)

    # seperate seg.
    prev_prob_map = infer(prev_img_file, model)
    curr_prob_map = infer(curr_img_file, model)

    # single cd
    prob_l2 = np.linalg.norm(prev_prob_map-curr_prob_map, axis=2, ord=2, keepdims=True)
    norm_prob_l2 = (prob_l2 - np.min(prob_l2)) / (np.max(prob_l2) - np.min(prob_l2))

    cos_dis_uint8 = (norm_prob_l2 * 255).astype(np.uint8)
    # collect non-zero results.
    if i == 0:
        conc_df = cos_dis_uint8[cos_dis_uint8 > 0]
    else:
        conc_df = np.concatenate([conc_df, cos_dis_uint8[cos_dis_uint8>0]], axis=0)
    '''
    Save info.
    '''
    res_dict[cd_cosdis_name] = {}
    res_dict[cd_cosdis_name]['dis'] = cos_dis_uint8

    
    '''
    Save cos-dis map.        
    '''
    out_cos_path = os.path.join(out_dir, 'cosdis', cd_cosdis_name)
    cv2.imwrite(out_cos_path, cos_dis_uint8)

thres = otsu_thres(conc_df)
for img_name, img_dict in res_dict.items():
    cos_dis_uint8 = img_dict['dis']
    df_int8 = np.where(cos_dis_uint8>=thres, 255, 0).astype(np.uint8)
    out_png_path = os.path.join(out_dir, '255', img_name)
    cv2.imwrite(out_png_path, df_int8)