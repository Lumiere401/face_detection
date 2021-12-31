from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import pandas as pd
import cv2
import argparse
from PIL import Image

from config import cfg
from core.net.network import Net
import torch
import torch.nn as nn
import torch.optim as optim


def main(cfg):
    to_tensor = T.Compose([
        T.Resize(args.input_train_size, interpolation=3),
        T.RandomHorizontalFlip(p=args.input_prob),
        T.Pad(args.input_padding),
        T.RandomCrop(args.input_train_size),
        T.ToTensor(),
        T.Normalize(mean=args.input_PIXEL_MEAN, std=args.input_PIXEL_STD),
    ])

    face_dataset = FaceDetectDataset(csv_file=args.csv_file,
                                     root_dir=args.train_path, transform=to_tensor)

    trainloader = torch.utils.data.DataLoader(face_dataset, batch_size=20,
                                              shuffle=True, num_workers=2)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = Net(cfg.pid_num)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(cfg.total_train_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0
        if epoch % 10 == 0:  # print every 10 epoch save the model
            file_path = os.path.join(cfg.output_path, 'model_{}.pkl'.format(epoch))
            torch.save(net.state_dict(), file_path)

    print('Finished Training')



def _load_images_path(folder_dir):
    '''
    :param folder_dir:
    :return: [(path, identiti_id, camera_id)]
    '''
    samples = []
    for root_path, _, files_name in os.walk(folder_dir):
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id = int(file_name.split('.')[0].split('_')[-1])
                full_path = root_path + os.sep + file_name
                samples.append([full_path, identi_id])
    return samples

class FaceDetectDataset(Dataset):
    """面部标记数据集."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file（string）：带注释的csv文件的路径。
        root_dir（string）：包含所有图像的目录。
        transform（callable， optional）：一个样本上的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = _load_images_path(self.root_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = self.dataset[idx][0]
        label = self.dataset[idx][1]
        image = cv2.imread(img_name)
        image = Image.fromarray(image)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample['image'], sample['label']


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

    main(cfg)




