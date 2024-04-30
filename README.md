# Facial-Expression-Recognition
This project contains a convolutional neural network (CNN) for recognizing emotions from facial expressions. The model is built using TensorFlow 1.x and is designed to classify images into one of seven emotion categories: angry, disgusted, fearful, happy, sad, surprised, and neutral.
## Prerequisites
Before running this project, you will need the following:

- Python 3.x

- TensorFlow 1.x

- NumPy

- OpenCV

Ensure that you have a compatible version of TensorFlow for TensorFlow 1.x functionalities.


## ACNN-ALSTM
The idea for this model comes from this article： https://www.hindawi.com/journals/cin/2022/7450637/
The ACNN-ALSTM model proposed in the article consists mainly of five parts：
CNN Local Feature Extraction Layer, Local Feature Attention Layer, LSTM Global Feature Learning Layer, Global Feature Attention Layer and Classification Layer

<img width="718" alt="Screenshot 2024-04-30 at 4 15 24 PM" src="https://github.com/Chengxin-Wu/Facial-Expression-Recognition/assets/112346517/cab365cd-ec45-4dc2-9aff-3dc702696428">


By calculating the attention distribution, the model can identify and focus on the key
features that best represent facial expressions. This helps the model to recognize and
classify facial expressions more accurately.

Local Attention Layer and Global Feature Attention Layer perform roughly the same tasks, but for different reasons. Calculatinng the correlation between local features and the query vector helps the model focus on important local areas in an image, while calculating the correlation between hidden states and the query vector helps the model focus on critical moments or historical information in time series data. The difference between them lies in the focus: one targets local features in images, while the other targets hidden states in time series data.
