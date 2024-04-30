# Facial-Expression-Recognition
This project contains a convolutional neural network (CNN) for recognizing emotions from facial expressions. The model is built using TensorFlow 1.x and is designed to classify images into one of seven emotion categories: angry, disgusted, fearful, happy, sad, surprised, and neutral.
## Prerequisites
Before running this project, you will need the following:

- Python 3.x

- TensorFlow 1.x

- NumPy

- OpenCV

Ensure that you have a compatible version of TensorFlow for TensorFlow 1.x functionalities.
## Dataset
Dataset website:https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

We chose the widely used dataset: the FER-2013 dataset. It features over 35 thousand images of human faces, and each image within the dataset is 48 by 48 pixels, presented in grayscale. Also, the dataset categorizes emotions into 7 distinct labels: They are anger, disgust, fear, happiness, sadness, surprise and neutrality. To ensure our data is ready for training, we implemented some preprocess techniques. First we know that Each entry in the 'pixels' column of our dataset is a string of pixel values, which represent a face. We split these strings into individual pixel values and convert them to integers. Then, we reshape this list of pixels into a two-dimensional 48 by 48 pixel array. After reshaping the images,we add a channel dimension to the data. Even though our images are grayscale and only need one channel, most deep learning frameworks require data to be in the form of samples, height, width, and channels. We expand our data to fit this requirement, ensuring compatibility with convolutional neural network architectures. Lastly, we convert the emotion labels to the one-hot encoding format, which is helpful for our classification work.
## DNN


## ACNN-ALSTM
The idea for this model comes from this article： https://www.hindawi.com/journals/cin/2022/7450637/
The ACNN-ALSTM model proposed in the article consists mainly of five parts：
CNN Local Feature Extraction Layer, Local Feature Attention Layer, LSTM Global Feature Learning Layer, Global Feature Attention Layer and Classification Layer

<img width="718" alt="Screenshot 2024-04-30 at 4 15 24 PM" src="https://github.com/Chengxin-Wu/Facial-Expression-Recognition/assets/112346517/cab365cd-ec45-4dc2-9aff-3dc702696428">


By calculating the attention distribution, the model can identify and focus on the key
features that best represent facial expressions. This helps the model to recognize and
classify facial expressions more accurately.

Local Attention Layer and Global Feature Attention Layer perform roughly the same tasks, but for different reasons. Calculatinng the correlation between local features and the query vector helps the model focus on important local areas in an image, while calculating the correlation between hidden states and the query vector helps the model focus on critical moments or historical information in time series data. The difference between them lies in the focus: one targets local features in images, while the other targets hidden states in time series data.
