import cv2
import os
import pandas as pd
def img_rename(path):
    name = os.listdir(path)
    id = 0

    # 标签文件生成
    dict = {'name': name}
    df = pd.DataFrame(dict)
    df.to_csv('person.csv')

    for files in name:
        num = 0
        for img in os.listdir(os.path.join(path,files)):
            if 'jpg' in img:
                img = cv2.imread(os.path.join(path,files,img))
            else:
                continue
            new_path = os.path.join(path, files, "%03d"%num + '_' + "%02d"%id)   #按后两位作为类别
            num += 1
            cv2.imwrite(new_path + '.jpg',img)
        id += 1








if __name__ == '__main__':
    path = './data'
    img_rename(path)