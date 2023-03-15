import cv2
import numpy as np
import os

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = 255 - image  # 反轉顏色
    image = image / 255.0  # 正規化
    return image

image_folder = './hand_write'
image_files = os.listdir(image_folder)
num_images = len(image_files)

images = np.zeros((num_images, 28, 28))
labels = np.zeros(num_images, dtype=np.int32)

for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    image = preprocess_image(image_path)
    label = int(os.path.splitext(image_file)[0].split('-')[0])  # 假設檔名是標籤，例如：5.png
    
    images[i] = image
    labels[i] = label

# 儲存圖像和標籤為 NumPy 數組
np.savez('my_mnist.npz', images=images, labels=labels)
