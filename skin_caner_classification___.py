
import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm
import os
import cv2
import re
import matplotlib.pyplot as plt
from glob import glob
import tensorflow_hub as hub
import tensorflow as tf
import time
from PIL import Image
from scipy.spatial.qhull import QhullError
from scipy import spatial
spatial.QhullError = QhullError
from imgaug import augmenters as iaa
import os
import cv2
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip

# import imgaug as ia
# import imgaug.augmenters as iaa
from tensorflow.keras.utils import load_img
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from albumentations import RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, ShiftScaleRotate, Rotate

hp = {}
hp['image_size'] = 512
hp['num_channels'] = 3
hp['batch_size'] = 32
hp['lr'] = 1e-4
hp["num_epochs"] = 30
hp['num_classes'] = 8
hp['dropout_rate'] = 0.1
hp['class_names'] = ["MEL", "SCC", "BCC", "AK", "BKL", "DF", "VASC", "NV"]

#read data
md = pd.read_csv("C://Users//mahid//Downloads//isic-2019//ISIC_2019_Training_GroundTruth.csv")
md.head()

md.shape

# seperate Melenoma, Basal cell carcinoma, Squamous cell carcinoma from dataset
mel_images = md.loc[md['MEL'] == 1, 'image'].tolist()
bcc_images = md.loc[md['BCC'] == 1, 'image'].tolist()
scc_images = md.loc[md['SCC'] == 1, 'image'].tolist()

nv_images = md.loc[md['NV'] == 1, 'image'].tolist()
ak_images = md.loc[md['AK'] == 1, 'image'].tolist()
bkl_images = md.loc[md['BKL'] == 1, 'image'].tolist()

vasc_images = md.loc[md['VASC'] == 1, 'image'].tolist()
unk_images = md.loc[md['UNK'] == 1, 'image'].tolist()
df_images = md.loc[md['DF'] == 1, 'image'].tolist()

# length of data
mel_count = len(mel_images)
bcc_count = len(bcc_images)
scc_count = len(scc_images)

nv_count = len(nv_images)
ak_count = len(ak_images)
bkl_count = len(bkl_images)

vasc_count = len(vasc_images)
unk_count = len(unk_images)
df_count = len(df_images)

print(mel_count, bcc_count, scc_count, nv_count, ak_count, bkl_count, vasc_count, df_count)

labels =["MEL", "SCC", "BCC", "AK", "BKL", "DF", "VASC", "NV"]
sizes = [mel_count, bcc_count, scc_count, ak_count, bkl_count,df_count,vasc_count, nv_count ]

#plot pie chart
colors = ['#ff9999', '#66b3ff', '#99ff99']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.title('Distribution of Images')

MEL = []
SCC = []
BCC = []
NV = []
AK = []
VASC = []
DF = []
BKL = []

path = "C://Users//mahid//Downloads//isic-2019//ISIC_2019_Training_Input//ISIC_2019_Training_Input"
for i in os.listdir(path):
    #print(i)
    name = i.split('.')[-2]
    if name in mel_images:
        MEL.append(os.path.join(path, i))
    elif name in scc_images:
        SCC.append(os.path.join(path, i))
    elif name in bcc_images:
        BCC.append(os.path.join(path, i))
    elif name in nv_images:
        NV.append(os.path.join(path, i))
    elif name in ak_images:
        AK.append(os.path.join(path, i))
    elif name in vasc_images:
        VASC.append(os.path.join(path, i))
    elif name in df_images:
        DF.append(os.path.join(path, i))
    elif name in bkl_images:
        BKL.append(os.path.join(path, i))

len(MEL), len(SCC), len(BCC), len(NV), len(AK), len(VASC), len(DF), len(BKL)

"""# Data Preprocessing

# Hair removing Function
"""

def apply_dullrazor(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Black hat filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)

    # Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)

    # Binary thresholding (MASK)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)

    # Replace pixels of the mask
    dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)

    return img, dst

original, processed = apply_dullrazor(MEL[100])

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title('Original Dermoscopy Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed)
plt.title('Segmented Image')
plt.axis('off')

original_images = []
processed_images = []

os.makedirs('C:/Users/mahid/Downloads/isic-2019//preproessed')

# len(MEL), len(SCC), len(BCC), len(NV), len(AK), len(VASC), len(DF), len(BKL)
img_list = MEL[90:95] + SCC[90:95] + BCC[90:95] + NV[90:95] + AK[90:95] + VASC[90:95]+DF[90:95] + BKL[90:95]
len(img_list)

for i, filename in enumerate(img_list):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        cv2.imwrite("/kaggle/working/preproessed/img_"+str(i)+".png", processed)

        # Append to the lists
        original_images.append(original)
        processed_images.append(processed)

fig, axs = plt.subplots(len(original_images), 2, figsize=(10, 5 * len(original_images)))

for i in range(len(original_images)):
    axs[i, 0].imshow(original_images[i])
    axs[i, 0].set_title('Original Image'+str([i]))
    axs[i, 0].axis('off')

    axs[i, 1].imshow(processed_images[i])
    axs[i, 1].set_title('Processed Image')
    axs[i, 1].axis('off')

"""# Apply Hair removal to all images"""

os.makedirs("C:/Users/mahid/Downloads/isic-2019/DF", exist_ok=True)

import shutil
shutil.rmtree("C:/Users/mahid/Downloads/isic-2019//DF")

os.makedirs("C:/Users/mahid/Downloads/isic-2019//DF/MEL", exist_ok=True)
os.makedirs("C:/Users/mahid/Downloads/isic-2019//DF/SCC", exist_ok=True)
os.makedirs("C:/Users/mahid/Downloads/isic-2019//DF/BCC", exist_ok=True)

# len(MEL), len(SCC), len(BCC), len(NV), len(AK), len(VASC), len(DF), len(BKL)

os.makedirs("C:/Users/mahid/Downloads/isic-2019//DF/NV", exist_ok=True)
os.makedirs("C:/Users/mahid/Downloads/isic-2019//DF/AK", exist_ok=True)
os.makedirs("C:/Users/mahid/Downloads/isic-2019//DF/VASC", exist_ok=True)

os.makedirs("C:/Users/mahid/Downloads/isic-2019//DF/DF", exist_ok=True)
os.makedirs("C:/Users/mahid/Downloads/isic-2019//DF/BKL", exist_ok=True)
# os.makedirs("/kaggle/working/DA/BCC", exist_ok=True)

start = time.time()
for i, filename in enumerate(MEL):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        processed = cv2.resize(processed, (512,512))
        cv2.imwrite("C:/Users/mahid/Downloads/isic-2019//DF/MEL/img_"+str(i)+".png", processed)
print("Time Taken: %f" % (time.time() - start))

start = time.time()
for i, filename in enumerate(SCC):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        processed = cv2.resize(processed, (512,512))
        cv2.imwrite("C:/Users/mahid/Downloads/isic-2019/DF/SCC/img_"+str(i)+".png", processed)
print("Time Taken: %f" % (time.time() - start))

start = time.time()
for i, filename in enumerate(BCC):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        processed = cv2.resize(processed, (512,512))
        cv2.imwrite("C:/Users/mahid/Downloads/isic-2019//DF/BCC/img_"+str(i)+".png", processed)

print("Time Taken: %f" % (time.time() - start))

# len(MEL), len(SCC), len(BCC), len(NV), len(AK), len(VASC), len(DF), len(BKL)
start = time.time()
for i, filename in enumerate(NV):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        processed = cv2.resize(processed, (512,512))
        cv2.imwrite("C:/Users/mahid/Downloads/isic-2019/DF/NV/img_"+str(i)+".png", processed)
print("Time Taken: %f" % (time.time() - start))

start = time.time()
for i, filename in enumerate(AK):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        processed = cv2.resize(processed, (512,512))
        cv2.imwrite("C:/Users/mahid/Downloads/isic-2019/DF/AK/img_"+str(i)+".png", processed)
print("Time Taken: %f" % (time.time() - start))

start = time.time()
for i, filename in enumerate(VASC):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        processed = cv2.resize(processed, (512,512))
        cv2.imwrite("C:/Users/mahid/Downloads/isic-2019/DF/VASC/img_"+str(i)+".png", processed)

print("Time Taken: %f" % (time.time() - start))

start = time.time()
for i, filename in enumerate(DF):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        processed = cv2.resize(processed, (512,512))
        cv2.imwrite("C:/Users/mahid/Downloads/isic-2019/DF/DF/img_"+str(i)+".png", processed)
print("Time Taken: %f" % (time.time() - start))

start = time.time()
for i, filename in enumerate(BKL):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Apply the function
        original, processed = apply_dullrazor(filename)
        processed = cv2.resize(processed, (512,512))
        cv2.imwrite("C:/Users/mahid/Downloads/isic-2019/DF/BKL/img_"+str(i)+".png", processed)


print("Time Taken: %f" % (time.time() - start))

"""#  Data Augmentation"""

import os
import cv2
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip

def augment_data_bcc(images, save_path, W=224, H=224, augment=True):
    save_images = []
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists

    for x in tqdm(images, total=len(images)):
        name = x.split("/")[-1].split(".")
        image_name = name[0]
        image_ext = name[1]

        image = cv2.imread(x)

        if augment:  # Augmentations applied if augment is True
            # Apply HorizontalFlip
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image)
            x1 = augmented["image"]

            # Apply VerticalFlip
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=image)
            x2 = augmented["image"]

            aug = Rotate(p=1.0, limit=270)
            augemented = aug(image=image)
            x3 = augemented["image"]

            aug = Rotate(p=1.0, limit=90)
            augemented = aug(image=image)
            x4 = augemented["image"]

            # Collect original and augmented images
            save_images = [(image, "original"), (x1, "aug1"), (x2, "aug2"), (x3,"aug3"), (x4, "aug4")]
        # else:
        #     save_images = [(image, "original")]  # Only original image

        try:
            # Save all augmented images with unique filenames
            for img, suffix in save_images:
                img_resized = cv2.resize(img, (W, H))  # Resize to (224, 224)
                temp_img_name = f"{image_name}_{suffix}.{image_ext}"  # Add suffix to filename
                image_path = os.path.join(save_path, temp_img_name)
                cv2.imwrite(image_path, img_resized)
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            continue



def augment_data_bcc(images, save_path, W=224, H=224, augment=True):
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists

    for x in tqdm(images, total=len(images)):
        name = os.path.basename(x).split(".")  # Better handling for Windows paths
        image_name = name[0]
        image_ext = name[1]

        image = cv2.imread(x)

        if image is None:
            print(f"Failed to read image: {x}")
            continue  # Skip the image if reading fails

        save_images = [(image, "original")]  # Start with the original image

        if augment:  # Augmentations applied if augment is True
            # Apply HorizontalFlip
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image)
            x1 = augmented["image"]

            # Apply VerticalFlip
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=image)
            x2 = augmented["image"]

            # Collect original and augmented images
            save_images.extend([(x1, "aug1"), (x2, "aug2")])

        try:
            # Save all images (original and augmented) with unique filenames
            for img, suffix in save_images:
                img_resized = cv2.resize(img, (W, H))  # Resize to (W, H)
                temp_img_name = f"{image_name}_{suffix}.{image_ext}"  # Add suffix to filename
                image_path = os.path.join(save_path, temp_img_name)
                cv2.imwrite(image_path, img_resized)
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            continue

# Example usage
augment_data_bcc(BCC, "C:/Users/mahid/Downloads/isic-2019/DAA/BCC")

bcc_l = glob("C:/Users/mahid/Downloads/isic-2019/DAA/BCC/*")
len(bcc_l)

import os
import cv2
from tqdm import tqdm
from albumentations import HorizontalFlip

def augment_data_mel(images, save_path, W=224, H=224, augment=True):
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists

    for x in tqdm(images, total=len(images)):
        name = os.path.basename(x).split(".")  # Use os.path.basename for better path handling
        image_name = name[0]
        image_ext = name[1]

        image = cv2.imread(x)

        if image is None:
            print(f"Failed to read image: {x}")
            continue  # Skip the image if reading fails

        save_images = [(image, "original")]  # Start with the original image

        if augment:  # Apply augmentations if augment is True
            # Apply Horizontal Flip
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image)
            x1 = augmented["image"]

            # Collect original and augmented images
            save_images.append((x1, "aug1"))

        try:
            # Save all images (original and augmented) with unique filenames
            for img, suffix in save_images:
                img_resized = cv2.resize(img, (W, H))  # Resize to (W, H)
                temp_img_name = f"{image_name}_{suffix}.{image_ext}"  # Add suffix to filename
                image_path = os.path.join(save_path, temp_img_name)
                cv2.imwrite(image_path, img_resized)  # Save the image
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            continue

# Example usage
augment_data_mel(MEL, "C:/Users/mahid/Downloads/isic-2019/DA/MEL")

mel_l = glob("C:/Users/mahid/Downloads/isic-2019/DA/MEL/*")
len(mel_l)

import os
import cv2
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip

def augment_data_bcc(images, save_path, W=224, H=224, augment=True):
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists

    for x in tqdm(images, total=len(images)):
        name = os.path.basename(x).split(".")  # Better handling for Windows paths
        image_name = name[0]
        image_ext = name[1]

        image = cv2.imread(x)

        if image is None:
            print(f"Failed to read image: {x}")
            continue  # Skip the image if reading fails

        save_images = [(image, "original")]  # Start with the original image

        if augment:  # Augmentations applied if augment is True
            # Apply HorizontalFlip
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image)
            x1 = augmented["image"]

            # Apply VerticalFlip
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=image)
            x2 = augmented["image"]

            # aug = Rotate(p=1.0, limit=270)
            # augemented = aug(image=image)
            # x3 = augemented["image"]

            # aug = Rotate(p=1.0, limit=90)
            # augemented = aug(image=image)
            # x4 = augemented["image"]

            # Collect original and augmented images
            save_images.extend([(x1, "aug1"), (x2, "aug2")]) # , (x3, "aug3"), (x4, "aug4")

        try:
            # Save all images (original and augmented) with unique filenames
            for img, suffix in save_images:
                img_resized = cv2.resize(img, (W, H))  # Resize to (W, H)
                temp_img_name = f"{image_name}_{suffix}.{image_ext}"  # Add suffix to filename
                image_path = os.path.join(save_path, temp_img_name)
                cv2.imwrite(image_path, img_resized)
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            continue

# Example usage
#augment_data_bcc(AK, "C:/Users/mahid/Downloads/isic-2019/DAA/AK")

ak_l = glob("C:/Users/mahid/Downloads/isic-2019/DAA/AK/*")
len(ak_l)

augment_data_bcc(SCC, "C:/Users/mahid/Downloads/isic-2019/DAA/SCC")

augment_data_bcc(DF, "C:/Users/mahid/Downloads/isic-2019/DAA/DF")

augment_data_bcc(VASC, "C:/Users/mahid/Downloads/isic-2019/DAA/VASC")

augment_data_bcc(NV, "C:/Users/mahid/Downloads/isic-2019/DAA/NV")

augment_data_bcc(BKL, "C:/Users/mahid/Downloads/isic-2019/DAA/BKL")

"""## After Augmentation

## Balanced Data
"""

import os
import shutil

def move_images_from_folders(src_folder, dest_folder, image_limit=1195):
    # Get all subfolders from the source folder
    subfolders = [folder for folder in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, folder))]

    for folder in subfolders:
        # Create a corresponding subfolder in the destination folder
        new_dest_folder = os.path.join(dest_folder, folder)
        os.makedirs(new_dest_folder, exist_ok=True)

        # Get all images in the current subfolder
        folder_path = os.path.join(src_folder, folder)
        images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]

        # Sort images to ensure consistent ordering
        images.sort()

        # Take only the first `image_limit` images
        selected_images = images[:image_limit]

        for img_path in selected_images:
            # Extract image file name
            img_name = os.path.basename(img_path)

            # Define the new path in the destination subfolder
            new_img_path = os.path.join(new_dest_folder, img_name)

            try:
                # Copy the image to the new destination subfolder (use `shutil.move` to move instead of copy)
                shutil.copy(img_path, new_img_path)
                print(f"Copied {img_name} to {new_dest_folder}")
            except Exception as e:
                print(f"Failed to copy {img_name}: {e}")

# Example usage
src_folder = "C:/Users/mahid/Downloads/isic-2019/DAA"
dest_folder = "C:/Users/mahid/Downloads/selected_images_new"
move_images_from_folders(src_folder, dest_folder)

ak =  glob("C:/Users/mahid/Downloads/selected_images_new/AK/*")
mel = glob("C:/Users/mahid/Downloads/selected_images_new/MEL/*")
bcc = glob("C:/Users/mahid/Downloads/selected_images_new/BCC/*")
scc = glob("C:/Users/mahid/Downloads/selected_images_new/SCC/*")
nv =  glob("C:/Users/mahid/Downloads/selected_images_new/NV/*")
bkl = glob("C:/Users/mahid/Downloads/selected_images_new/BKL/*")
df =  glob("C:/Users/mahid/Downloads/selected_images_new/DF/*")
vasc =glob("C:/Users/mahid/Downloads/selected_images_new/VASC/*")

new_mel_count = len(mel)
new_bcc_count = len(bcc)
new_scc_count = len(scc)
new_nv_count = len(nv)
new_ak_count = len(ak)
new_bkl_count = len(bkl)
new_df_count = len(df)
new_vasc_count = len(vasc)

labels = ["MEL", "BCC", "SCC", "AK", "BKL", "DF", "VASC", "NV" ]
sizes = [new_mel_count, new_bcc_count, new_scc_count, new_ak_count,new_bkl_count,  new_df_count, new_vasc_count,new_nv_count]

# Colors for the pie chart (you can expand the color list if needed)
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#c4e17f']

# Plot the pie chart
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

# Adding a central circle for a donut chart effect
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')
plt.title('Distribution of Images across 8 Classes')
plt.show()