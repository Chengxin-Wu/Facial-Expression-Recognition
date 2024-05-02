# Facial-Expression-Recognition
This project contains a convolutional neural network (CNN) for recognizing emotions from facial expressions. The model is built using TensorFlow 1.x and is designed to classify images into one of seven emotion categories: angry, disgusted, fearful, happy, sad, surprised, and neutral.
## Group member
Chengxin Wu NetID: CW1171

Yuanhao Jiang NetID: YJ270
## Prerequisites
Before running this project, you will need the following:

- Python 3.x
- TensorFlow 1.x
- NumPy
- OpenCV

Ensure that you have a compatible version of TensorFlow for TensorFlow 1.x functionalities.
## How to run our code
- Download dataset from Kaggle link below
- Change data dir to your own
- Run different files and get results
- For camera_test.py, use
```python
python camera_test.py
```

## Dataset
Dataset website: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data.

It features over 35 thousand pixels of human faces, and each image within the dataset is 48 by 48 pixels, presented in grayscale. Also, the dataset categorizes emotions into 7 distinct labels: They are anger, disgust, fear, happiness, sadness, surprise and neutrality. 
## DNN
We chose the DNN model from this article: https://www.cnblogs.com/XDU-Lakers/p/10587894.html as a comprision of our model.
The DNN architecture consists of the following layers:

- **Input Layer**: The input images are reshaped to 48x48 pixels with a single color channel (grayscale).
- **First Convolutional Layer**: Consists of 64 filters of size 5x5, followed by ReLU activation and a max pooling operation. Local response normalization is applied post-pooling.
- **Second Convolutional Layer**: Utilizes 64 filters of size 3x3, followed by ReLU activation. Local response normalization and another max pooling step follow.
- **Fully Connected Layer 1**: A dense layer with 384 units, followed by ReLU activation.
- **Fully Connected Layer 2**: Another dense layer with 192 units.
- **Output Layer**: A linear layer with 7 units corresponding to the emotion categories.

Since the code structure is too old, we reconstructed it and make a more advanced structure. And the 



## ACNN-ALSTM
The idea for this model comes from this article： https://www.hindawi.com/journals/cin/2022/7450637/
The ACNN-ALSTM model proposed in the article consists mainly of five parts：
CNN Local Feature Extraction Layer, Local Feature Attention Layer, LSTM Global Feature Learning Layer, Global Feature Attention Layer and Classification Layer

<img width="718" alt="Screenshot 2024-04-30 at 4 15 24 PM" src="https://github.com/Chengxin-Wu/Facial-Expression-Recognition/assets/112346517/cab365cd-ec45-4dc2-9aff-3dc702696428">


By calculating the attention distribution, the model can identify and focus on the key
features that best represent facial expressions. This helps the model to recognize and
classify facial expressions more accurately.

Local Attention Layer and Global Feature Attention Layer perform roughly the same tasks, but for different reasons. Calculatinng the correlation between local features and the query vector helps the model focus on important local areas in an image, while calculating the correlation between hidden states and the query vector helps the model focus on critical moments or historical information in time series data. The difference between them lies in the focus: one targets local features in images, while the other targets hidden states in time series data.

Test Model


<img width="564" alt="Screenshot 2024-04-30 at 5 00 45 PM" src="https://github.com/Chengxin-Wu/Facial-Expression-Recognition/assets/112346517/ba36df35-bfc4-4dfb-913b-3beba0085d35">


## Camera Test
The method adopted involves using a camera to recognize facial micro-expressions in real-time. The specific process is as follows: use OpenCV's built-in face detector, capture facial images using the system camera, preprocess the facial images, input the processed images into the model, and finally display the results analyzed by the model in the running window. 

The results is as follows: 
![image](https://github.com/Chengxin-Wu/Facial-Expression-Recognition/assets/48239248/9799d50e-2b1b-4a44-bdac-276fd8175863)
## Slide link
https://docs.google.com/presentation/d/1eOU_f3yD8aZDESqQQU7q_8-ZDfPiVXbNaoAtc-vJmW0/edit#slide=id.p1

## Additional Work After Presentation
- Improved the model by adding activation functions and regularization to alleviate model overfitting, but due to the lack of significant improvement, these were not included in the final submitted model.
- Try different LSTM model parameters to observe their impact on the model.
- Use more data to evaluate the model's performance

## Summary of work we did
- We reconstructed the DNN model and added the test part.
- Build the ACC-ALSTM model based on the description of its architecture in the article, while also constructing the CNN-ALSTM and CNN-LSTM models for comparison.
- Choose an appropriate loss function, optimizer, batch size, learning rate, and other hyperparameters to train the model
- Use our model for the video recording function on phones or computers to recognize the facial expressions of people appearing on camera
