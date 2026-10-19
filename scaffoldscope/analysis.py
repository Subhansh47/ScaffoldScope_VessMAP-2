import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label

def analyze(mask, pixel_size=None, min_object_pixels=8):
    binary=np.asarray(mask,dtype=bool)
    clean=remove_small_objects(binary,min_size=int(min_object_pixels)) if binary.any() else binary
    skel=skeletonize(clean); p=np.pad(skel,1); h,w=skel.shape; degree=np.zeros((h,w),np.uint8)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx or dy: degree += p[1+dy:1+dy+h,1+dx:1+dx+w]
    endpoints=int(np.sum(skel & (degree==1))); components=int(label(clean,connectivity=2).max())
    length_px=float(np.sum(skel & (degree>0))); scale=float(pixel_size) if pixel_size and pixel_size>0 else None
    return {'area_pixels':int(clean.sum()),'components':components,'endpoints':endpoints,'skeleton_pixels':int(skel.sum()),'approx_length_pixels':length_px,'area_scaled':clean.sum()*scale*scale if scale else None,'approx_length_scaled':length_px*scale if scale else None,'mask':clean,'skeleton':skel}
