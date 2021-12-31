import sys
import cv2
import torch
import argparse
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from core.net.network import Net
from config import cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="config/config.yml", help="path to config file", type=str
    )
    parser.add_argument("--csv_file", type=str, default='./data/person.csv')
    # parser.add_argument('--mode', type=str, default='test', help='train, test or visualize')
    parser.add_argument('--mode', type=str, default='train', help='train, test or visualize')
    parser.add_argument('--train_path', type=str, default='data/', help='path for the train image')
    parser.add_argument('--input_train_size', default=[224, 224])
    parser.add_argument('--input_prob', default=0.5)
    parser.add_argument('--input_padding', default=10)
    parser.add_argument('--input_PIXEL_MEAN', default=[0.485, 0.456, 0.406])
    parser.add_argument('--input_PIXEL_STD', default=[0.229, 0.224, 0.225])

    args = parser.parse_args()
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    to_tensor = T.Compose([
        T.Resize(args.input_train_size, interpolation=3),
        T.RandomHorizontalFlip(p=args.input_prob),
        T.Pad(args.input_padding),
        T.RandomCrop(args.input_train_size),
        T.ToTensor(),
        T.Normalize(mean=args.input_PIXEL_MEAN, std=args.input_PIXEL_STD),
    ])

    # 加载模型
    PATH = './results/model_40.pkl'
    model = Net(class_num=4)
    model.to(device)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    # 人脸识别分类器本地存储路径
    cascade_path = "/opt/anaconda3/envs/pytorch1.8/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"

    name = pd.read_csv(args.csv_file).values
    # 循环检测识别人脸
    while True:
        ret, frame = cap.read()  # 读取一帧视频

        if ret is True:

            # 图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                # 截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                image = to_tensor(Image.fromarray(image)).unsqueeze(0)
                output = model(image)
                output = output[0]
                output = output.tolist()
                faceID = output.index(max(output))
                print("faceID", faceID)
                # 如果是“我”
                if faceID >= 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    # 文字提示是谁
                    cv2.putText(frame, name[faceID][1],
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.putText(frame, 'Nobody',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                2,  # 字号
                                (255, 0, 0),  # 颜色
                                2)  # 字的线宽
                    pass

        cv2.imshow("Face Recognition", frame)

        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()