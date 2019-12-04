# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from IPython.display import Image
from keras.applications import imagenet_utils
import matplotlib.image as mpimg




# In[2]:


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(5,activation='softmax')(x) #final layer with softmax activation


# In[3]:


model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


# In[4]:


for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


# In[5]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory("C:/Users/Harishankar/Downloads/fixtures_photos",
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


# In[33]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=60)


#-----------------------------------------------------------------------------------------
#mobile = keras.applications.mobilenet.MobileNet()
#def prepare_image(file):
    #img_path = "C:/Users/Harishankar/Downloads"
    #img = image.load_img(img_path + file, target_size=(224, 224))
    #img_array = image.img_to_array(img)
    #img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    #return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

#Image(filename='dandelion.jpg')

#preprocessed_image=prepare_image('dandelion.jpg')

#predictions = mobile.predict(preprocessed_image)

#results = imagenet_utils.decode_predictions(predictions)

#results
#-----------------------------------------------------------------------------------------

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()  

    return img_tensor
classes=train_generator.class_indices
print(classes)

img_path = "C:/Users/Harishankar/Downloads/chandelier.jpg"
new_image = load_image(img_path)
pred=model.predict(new_image)

#results = imagenet_utils.decode_predictions(preds)

#print(results)

print(pred)


print(pred.all())

cl=np.argmax(pred[0])
print(cl)

if(cl==0):
    print('Chandelier')
    img=mpimg.imread('your_image.png')
    imgplot = plt.imshow(img)
    plt.show()
elif(cl==1):
    print('Floor lamp')
    img=mpimg.imread('your_image.png')
    imgplot = plt.imshow(img)   
    plt.show()
elif(cl==2):
    print('Pendant lamp')
    img=mpimg.imread('your_image.png')
    imgplot = plt.imshow(img)
    plt.show()
elif(cl==3):
    print('Table lamp')
    img=mpimg.imread("C:/Users/Harishankar/Downloads/fixtures_photos/table lamp/12.13084.jpg")
    imgplot = plt.imshow(img)
    plt.show()
    img=mpimg.imread("C:/Users/Harishankar/Downloads/fixtures_photos/table lamp/29.flexi-study-lamp-500x500.jpg")
    imgplot = plt.imshow(img)
    plt.show()
elif(cl==4):
    print('Wall lamp')
    img=mpimg.imread('your_image.png')
    imgplot = plt.imshow(img)
    plt.show()




#test=np.array(new_image)
#pred = model.predict_classes(test)



#label=pred.argmax(axis=-1)
#for cls in training_generator.class_indices:
    #print(cls+": "+preds[0][training_generator.class_indices[cls]])
#print(label)







