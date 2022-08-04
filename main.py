import os
from tqdm import tqdm
import cv2
import SimpleITK as sitk
from PIL import Image


def slice(ori_path: str, pro_path: str):
    global num
    global end
    for path in os.listdir(ori_path):
        if path.find('mhd') >= 0:
            save_content = pro_path

            data_mhd = sitk.ReadImage(os.path.join(ori_path, path))

            scan = sitk.GetArrayFromImage(data_mhd)
            for i in tqdm(range(len(scan))):
                img = cv2.normalize(scan[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                img = Image.fromarray(img)
                if save_content == 'train_slice/low_dose_slice':
                    save_path = os.path.join(save_content, f'{i + num}.png')
                    img.save(save_path)
                    end = i + num
                if save_content == 'train_slice/normal_dose_slice':
                    save_path = os.path.join(save_content, f'{i + num - end - 1}.png')
                    img.save(save_path)
            num = num + 1 + i


num = 0

low_dose_path = 'train/low dose'
low_dose_slice_path = 'saved/low'
normal_dose_path = 'train/normal dose'
normal_dose_slice_path = 'saved/normal'
slice(low_dose_path, low_dose_slice_path)
slice(normal_dose_path, normal_dose_slice_path)
