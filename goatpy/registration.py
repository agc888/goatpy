from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def prepare_maldi_img(maldi_img0, scale_factor, threshold, sparsity=None):
    original_maldi_shp = maldi_img0.shape #array - 10uM 
    target_maldi_size = (original_maldi_shp[1]*scale_factor, original_maldi_shp[0]*scale_factor)
    target_maldi_shp = (target_maldi_size[1], target_maldi_size[0])
    
    #Binarize and resize
    #maldi_img = Image.fromarray((maldi_img0>threshold)|(maldi_img0<20)) #note Image and array dimensions are inverted
    maldi_img = Image.fromarray(maldi_img0>threshold)
    #if sparsity:
    #    sparsity = min(max(0.1, sparsity), 0.9)
    #    while ((np.sum(maldi_img)/np.size(maldi_img))>sparsity):
    #        threshold = threshold*0.9
    #        maldi_img = Image.fromarray(maldi_img0>threshold)
    #        print("SPARSITY: %.3f"%np.sum(maldi_img)/np.size(maldi_img))
    #print("SPARSITY: %.3f"%(np.sum(maldi_img)/np.size(maldi_img))) #maldi_img is image not array
    maldi_img = maldi_img.resize(target_maldi_size, resample=Image.Resampling.NEAREST)
    maldi_img_array = np.array(maldi_img, dtype=np.float32)
    #maldi_img_array = (2*np.array(maldi_img, dtype=np.float32))-1

    plt.imshow(maldi_img_array)
    plt.savefig('maldi-bin.png')
    plt.close()
    return maldi_img_array

def prepare_he_img(he_img, buffer=2000, he_ratio=1.97863, he_threshold=244):
    original_he_size = he_img.size
    target_he_size = (int(original_he_size[0]/he_ratio), int(original_he_size[1]/he_ratio))
    buffered_he_shp = (target_he_size[1]+buffer, target_he_size[0]+buffer)
    
    #HE - binarize
    he_img = he_img.resize(target_he_size, resample=Image.Resampling.NEAREST)
    he_img_channel = he_img.split()[2] #blue channel
    he_img_channel = np.asarray(he_img_channel)
    he_img_bin = he_img_channel<he_threshold 
    #he_img_bin = (he_img_channel<he_threshold) & (he_img_channel>40)

    plt.imshow(he_img_bin) 
    plt.savefig('he-bin.png')
    plt.close()

    he_img_bin = Image.fromarray(he_img_bin)
    return he_img_bin, buffered_he_shp, target_he_size

def prepare_fiducial(he_img, he_threshold):
    he_img_channel = he_img.split()[2]
    he_img_channel = np.asarray(he_img_channel)
    he_img_bin = he_img_channel<he_threshold
    he_img_bin = Image.fromarray(he_img_bin)
    return he_img_bin


def prepare_f_img(f_img, f_ratio, f_threshold):
    original_f_size = f_img.size
    target_f_size = (int(original_f_size[0]/f_ratio), int(original_f_size[1]/f_ratio))
    target_f_shp = (target_f_size[1], target_f_size[0])
    
    #F - binarize
    f_img = f_img.resize(target_f_size, resample=Image.Resampling.NEAREST)
    f_img_channel = f_img.split()[2] #blue channel (assuming RGB)
    f_img_channel = np.asarray(f_img_channel)
    f_img_bin = f_img_channel>f_threshold
 
    plt.imshow(f_img_bin)
    plt.savefig('f-bin.png')
    plt.close()

    f_img_bin = Image.fromarray(f_img_bin)
    return f_img_bin, target_f_shp, target_f_size

def run_registration(he_container, maldi_img_array, plot=False):
    print(he_container.shape, maldi_img_array.shape)
    out = cv.matchTemplate(he_container, maldi_img_array, cv.TM_CCOEFF_NORMED)
    idx = np.unravel_index(np.argmax(out), out.shape)
    print(idx)    
    if plot:
        plt.imshow(out)
        plt.colorbar()
        plt.arrow(idx[1]-200, idx[0]-200, 100, 100, color='red', width=10)
        plt.savefig('rego.png')
        plt.close()
    return idx, out

def optimize_he_rotation_to_maldi(he_img_bin, maldi_img_array, he_rotations, target_he_size, buffered_he_shp, buffer=2000):
    max_he_score = 0
    for he_rotation in he_rotations:
        he_img_rot = he_img_bin.rotate(he_rotation, expand=True)
        rot_size = (he_img_rot.size[0], he_img_rot.size[1])

        he_container = np.zeros(buffered_he_shp, dtype=np.float32)
        hbuff_x = (target_he_size[1] + buffer - rot_size[1])//2
        hbuff_y = (target_he_size[0] + buffer - rot_size[0])//2

        he_container[hbuff_x:-hbuff_x,hbuff_y:-hbuff_y] = he_img_rot
        idx, out = run_registration(he_container, maldi_img_array, plot=True)
        #score = (np.max(out)-np.mean(out))/np.std(out)
        score = np.max(out)
        if score>max_he_score:
            max_he_score = score
            max_he_rotation = he_rotation
            max_hbuff_x = hbuff_x
            max_hbuff_y = hbuff_y
            max_idx = idx
            max_he_container = he_container
        print(he_rotation, np.max(out), score)
    return max_he_container, max_idx, max_he_score, max_he_rotation, max_hbuff_x, max_hbuff_y

def optimize_f_rotation_to_he(he_container, f_img_bin, f_rotations):
    #f_container, f_idx, f_rotation
    max_f_score = 0
    for f_rotation in f_rotations:
        f_img_rot = np.array(f_img_bin.rotate(f_rotation, expand=False), dtype=np.float32)
        idx, out = run_registration(he_container, f_img_rot, plot=False)
        score = (np.max(out)-np.mean(out))/np.std(out)
        if score>max_f_score:
            max_f_score = score
            max_f_rotation = f_rotation
            max_idx = idx
        print("TOM", f_rotation, score)
    print("TOM2", max_f_score, max_idx, max_f_rotation)
    return max_idx, max_f_rotation

def optimize_fiducial_rotation(base_img, rot_img, rotations, he_threshold):
    max_f_score = 0
    base_img = np.array(base_img, dtype=np.float32)
    for f_rotation in rotations:
        #f_img_rot = np.array(rot_img.rotate(f_rotation, expand=False), dtype=np.float32)
        f_img_rot = rot_img.rotate(f_rotation, expand=False)
        f_img_rot_channel = np.array(f_img_rot.split()[2])
        f_img_rot = np.array(f_img_rot_channel<he_threshold, dtype=np.float32)

        #idx, out = run_registration(base_img, f_img_rot, plot=False)
        out = cv.matchTemplate(base_img, f_img_rot, cv.TM_CCOEFF_NORMED)
        idx = np.unravel_index(np.argmax(out), out.shape)
        #score = (np.max(out)-np.mean(out))/np.std(out)
        score = np.max(out)
        if score>max_f_score:
            max_f_score = score
            max_f_rotation = f_rotation
            max_idx = idx
            max_he_bin = f_img_rot
            plt.imshow(out)
            plt.colorbar()
            plt.arrow(idx[1]-200, idx[0]-200, 100, 100, color='red', width=10)
            plt.savefig('rego.png')
            plt.close()
        print("TOM", f_rotation, score, np.max(out))
    print("TOM2", max_f_score, max_idx, max_f_rotation)
    return max_idx, max_f_rotation, max_he_bin

def put_img_in_container(img, container_shp, start_x, end_x, start_y, end_y, rotation, dtype, expand=True):
    img_rot = np.asarray(img.rotate(rotation, expand=expand), dtype=dtype)
    print(img_rot.shape, end_x-start_x, end_x, start_x)
    container = np.zeros(container_shp, dtype=dtype)
    #if max_hbuff_x>0 and max_hbuff_y>0:
    container[start_x:end_x,start_y:end_y] = img_rot
    #else:
    #    container[max_hbuff_x:-max_hbuff_x,max_hbuff_y:-max_hbuff_y] = img_rot
    
    return container




