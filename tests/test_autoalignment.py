import goatpy as gp
import pickle

with open("/Users/andrewcauser/Documents/Griffith/data_objects/maldi_sd.pkl", "rb") as f:
    maldi_sd = pickle.load(f)

with open("/Users/andrewcauser/Documents/Griffith/data_objects/he_sd.pkl", "rb") as f:
    he_sd = pickle.load(f)


def convert_sd_to_nparray:


def normalize_maldi_img(maldi_img, mask=None, scale=False):
    #return maldi_img
    from scipy.ndimage import convolve
    import matplotlib.pyplot as plt
    kernel = np.ones((40, 40))    
    #fig, ax = plt.subplots(1,3, figsize=(10,30))
    #ax[0].matshow(maldi_img)

    if mask is not None:
        sumval = convolve(maldi_img*mask.astype(np.float32) , kernel, mode='reflect')
        normval = convolve(mask.astype(np.float32), kernel, mode='reflect')
        mean = sumval/normval
        
        #plt.matshow(normval)
        #plt.colorbar()
        #plt.show()
    else:
        mean = convolve(maldi_img, kernel, mode='reflect')/kernel.size
    if scale:
        squared_diff = (maldi_img - mean)**2
        std = np.sqrt(convolve(squared_diff, kernel, mode='reflect')/kernel.size)
        maldi_img = (maldi_img-mean)/std
    else:
        maldi_img = maldi_img - mean


    #ax[1].matshow(np.clip(maldi_img, 0, None))
    #ax[2].matshow(mean)
    #plt.colorbar()
    #plt.show()
    return maldi_img

from PIL import Image
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


target_resolution = 0.5
maldi_pixel_size = 10
maldi_threshold = 1. #slide4 80 slide5 50

sparsity = 0.3
scale_factor = int(maldi_pixel_size/target_resolution) #integer may be critical here

he_pixel_size = 0.2527 #2023
he_threshold = 200 #slide2 #THIS ONE
he_ratio=target_resolution/he_pixel_size

f_pixel_size = 0.5
f_threshold = 50
f_ratio = target_resolution/f_pixel_size

buffer = 2000

maldi_img = normalize_maldi_img(img_2d, scale=True)
maldi_mask = maldi_img<0.25 #binarized cell area based on local-contrast enhanced z-score
maldi_img_array = prepare_maldi_img(maldi_img, scale_factor, maldi_threshold, sparsity=sparsity)
