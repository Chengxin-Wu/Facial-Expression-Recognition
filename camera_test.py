import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_captor = cv2.VideoCapture(0)

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


model_path = 'cnn_alstm.pth'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 第四卷积层
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # LSTM层
        self.lstm = nn.LSTM(input_size=4608, hidden_size=128, num_layers=1, batch_first=True)

        # 全局注意力层
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classification_layer = nn.Linear(128, 7)  # 128 是上一步 LSTM 或注意力的输出维度，7 是目标类别的数量
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 调整输入张量的形状
        x = x.view(-1, 1, 48, 48)

        # 第一卷积层
        x = self.conv1(x)
        x = self.pool1(x)

        # 第二卷积层
        x = self.conv2(x)
        x = self.pool2(x)

        # 第三卷积层
        x = self.conv3(x)

        # 第四卷积层
        x = self.conv4(x)
        x = self.pool3(x)

        # 展平张量以进行 LSTM 层
        x = x.view(x.size(0), -1)

        # LSTM 部分
        x, _ = self.lstm(x.unsqueeze(1))

        # 全局注意力层
        attention_weights = self.attention(x)
        attention_weights = nn.functional.softmax(attention_weights, dim=1)
        context_vector = torch.sum(x * attention_weights, dim=1)

        # 分类层
        x = self.classification_layer(context_vector)

        # Softmax 层
        x = self.softmax(x)


        # 返回最后的上下文向量
        return x

model = CNN()
dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(dict)

transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize the image to the required input size of the model
    transforms.Grayscale(),  # Convert the image to grayscale (if the model expects grayscale images)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to the range [-1, 1]
])

def predict_image(model, image_path, transform):
    # Load and preprocess the image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get the available device
    # Move the input tensor to the appropriate device
    image_tensor = image_tensor.to(device)
    # Set the model to evaluation mode
    model.eval()
    # Move the model to the appropriate device
    model = model.to(device)
    # Perform the prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        print(EMOTIONS[predicted_class])
        return EMOTIONS[predicted_class]


def image_to_tensor(image):
    tensor = np.asarray(image).reshape(-1, 2304) * 1 / 255.0
    return tensor

def format_image(image):
    # image如果为彩色图：image.shape[0][1][2](水平、垂直像素、通道数)
    if len(image.shape) > 2 and image.shape[2] == 3:
        # 将图片变为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        # 调整scaleFactor参数的大小，可以增加识别的灵敏度，推荐1.1
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    # 如果图片中没有检测到人脸，则返回None
    if not len(faces) > 0:
        return None, None
    # max_are_face包含了人脸的坐标，大小
    max_are_face = faces[0]
    # 在所有人脸中选一张最大的脸
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face

    # 这两步可有可无
    face_coor = max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # 调整图片大小，变为48*48
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("problem during resize")
        return None, None

    return image, face_coor


emoji_face = []
reslut = None
while True:
    ret, frame = video_captor.read()
    detected_face, face_coor = format_image(frame)
    image_filename = "face.png"

    if face_coor is not None:
        cv2.imwrite('a.jpg', detected_face)
        # print("Success")
        cv2.imwrite(image_filename, detected_face)
        # tensor = image_to_tensor(detected_face)
        # predict = predict_image(model, image_filename, transform)
        emotion_prediction = predict_image(model, image_filename, transform)
        # 在原始帧中绘制预测结果
        x, y, w, h = face_coor
        cv2.putText(frame, emotion_prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.imshow('face', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

