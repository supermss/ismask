# 사전설치 라이브러리 
# 1) pip install opencv-python
# 2) pip install cvlib
# 3) pip install opencv-python tensorflow
# 3) pip install keras
# 4) pip install keras-applications
# 5) pip install scikit-learn
# 6) pip install matplotlib

import cv2
import cvlib                    as cv
import datetime
import os
import numpy                    as np
import matplotlib.pyplot        as plt
import tensorflow               as tf

from   keras                    import layers
from   keras                    import models
from   keras.preprocessing      import image
from   sklearn.model_selection  import train_test_split
from   keras.layers             import Dense, Flatten, BatchNormalization
from   keras.applications       import ResNet50
from   keras.applications       import imagenet_utils


path_dir1 = "./capture/nomask/"
path_dir2 = "./capture/mask/"
 
file_list1 = os.listdir(path_dir1) # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2) # path에 존재하는 파일 목록 가져오기
 
file_list1_num = len(file_list1)  #폴더의 갯수 체크  
file_list2_num = len(file_list2)  #폴더의 갯수 체크
 
file_num = file_list1_num + file_list2_num #폴더의 갯수 합계

num = 0
all_img = np.float32(np.zeros((file_num, 140, 140, 3))) 
all_label = np.float64(np.zeros((file_num, 1)))

#path_dir1 폴더의 파일목록에서 140x140크기의 이미지를 배열로 all_img배열로 저장 all_label 배열은 0:nomask로 저장
for img_name in file_list1:
    img_path = path_dir1+img_name
    img = image.image_utils.load_img(img_path, target_size=(140, 140))
  
    x = image.image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)    
    
    x = imagenet_utils.preprocess_input(x)
   
    all_img[num, :, :, :] = x
    
    all_label[num] = 0 # nomask
    num = num + 1

#path_dir2 폴더의 파일목록에서 140x140크기의 이미지를 배열로 all_img배열로 저장 all_label 배열은 1:mask로 저장
for img_name in file_list2:
    img_path = path_dir2+img_name
    img = image.image_utils.load_img(img_path, target_size=(140, 140))   

    x = image.image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)
    all_img[num, :, :, :] = x
    
    all_label[num] = 1 # mask
    num = num + 1
 
 
# 데이터셋 섞기(적절하게 훈련되게 하기 위함) 
n_elem = all_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
all_label = all_label[indices]
all_img = all_img[indices]
 
 
# 훈련셋 테스트셋 분할
num_train = int(np.round(all_label.shape[0]*0.8))
num_test = int(np.round(all_label.shape[0]*0.2))
 
train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train:, :, :, :] 
 
train_label = all_label[0:num_train]
test_label = all_label[num_train:]
 
 

# create the base pre-trained model
IMG_SHAPE = (140, 140, 3)
 
base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))
 
flatten_layer = Flatten()
dense_layer1 = Dense(128, activation='relu')
bn_layer1 = BatchNormalization()
dense_layer2 = Dense(1, activation=tf.nn.sigmoid)
 
model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer1,
        bn_layer1,
        dense_layer2,
        ])
 
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
 
history = model.fit(train_img, train_label, epochs=20, batch_size=16, validation_data = (test_img, test_label))
 
 
# save model
model.save("model.h5")
print("Saved model to disk")  

"""  Chart 시각화 ppt용 출력용 model 사용시 사용안함
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch_range = range(20)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.plot(epoch_range, accuracy, label='Training Accuaracy')
plt.plot(epoch_range, val_accuracy, label='Validation Accuaracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epoch_range, loss, label='Training Loss')
plt.plot(epoch_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

"""



