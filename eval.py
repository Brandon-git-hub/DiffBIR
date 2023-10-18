import os
import torch
import numpy
from PIL import Image
from einops import rearrange
from skimage.metrics import peak_signal_noise_ratio

from utils.metrics import calculate_psnr_pt, LPIPS
from utils.file import list_image_files, get_file_name_parts
from utils.image import CenterCrop


def convert_tensor(array):
    image = array[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    return 2.*image - 1.



def _eval():
    dir = 'inputs/Gopro-scene'
    blur_dir = os.path.join(dir, 'blur')
    deblur_dir = os.path.join(dir, 'deblur')
    sharp_dir = os.path.join(dir, 'sharp')
    for deblur_file_path in list_image_files(deblur_dir, follow_links=True):
        deblur_img = Image.open(deblur_file_path).convert("RGB")
        deblur_img_crop = CenterCrop(deblur_img, 512)
        _, file_name = os.path.split(deblur_file_path)
        # print(parent_path, stem, ext) # inputs/Gopro-scene/deblur 20 .png
        sharp_file_path = os.path.join(sharp_dir, file_name)
        sharp_img = Image.open(sharp_file_path).convert("RGB")
        sharp_img_crop = CenterCrop(sharp_img, 512)
        
        stem, ext = os.path.splitext(file_name)
        result_file_path = os.path.join('results/Gopro-scene/deblur', stem+'_0'+ext)
        result_img = Image.open(result_file_path).convert("RGB")
        
        sharp_numpy = numpy.array(sharp_img).astype(numpy.float32) / 255.0
        sharp_tensor = convert_tensor(sharp_numpy)
        sharp_numpy_crop = numpy.array(sharp_img_crop).astype(numpy.float32) / 255.0
        sharp_tensor_crop = convert_tensor(sharp_numpy_crop)
        
        deblur_numpy = numpy.array(deblur_img).astype(numpy.float32) / 255.0
        deblur_tensor = convert_tensor(deblur_numpy)
        deblur_numpy_crop = numpy.array(deblur_img_crop).astype(numpy.float32) / 255.0
        deblur_tensor_crop = convert_tensor(deblur_numpy_crop)
        result_numpy = numpy.array(result_img).astype(numpy.float32) / 255.0
        result_tensor = convert_tensor(result_numpy)

        print('='*15)
        print(file_name)
        psnr_reg = peak_signal_noise_ratio(sharp_numpy_crop, deblur_numpy_crop, data_range=1)
        psnr_diff = peak_signal_noise_ratio(sharp_numpy_crop, result_numpy, data_range=1) 
        print(f'PSNR reg: {psnr_reg}, PSNR diff: {psnr_diff}')

        lpips = LPIPS('alex')
        lpips_reg = lpips(sharp_tensor_crop, deblur_tensor_crop, False)
        lpips_diff = lpips(sharp_tensor_crop, result_tensor, False)
        print(f'LPIPS (Alex) reg: {lpips_reg}, diff: {lpips_diff}')

        if True:
            blur_file_path = os.path.join(blur_dir, file_name)
            blur_img = Image.open(blur_file_path).convert("RGB")
            blur_img_crop = CenterCrop(blur_img, 512)
            n_samples = numpy.concatenate([deblur_numpy_crop, result_numpy, sharp_numpy_crop], axis=1)
            n_samples = 255. * n_samples
            if True:
                n_samples = numpy.concatenate([numpy.array(blur_img_crop), n_samples], axis=1)
            Image.fromarray(n_samples.astype(numpy.uint8)).save('results/seq/'+file_name)

        


if __name__ == "__main__":
    _eval()