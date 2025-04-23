import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

#from src.model_yuanban import u2net_full
#from src.model_decoder import  u2net_full
#from att_unet import u2net_full
from DNANET import u2net_full
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# def main():  #原本的预测
#     #weights_path = "save_weights/u2net_decoder_100epoch.pth"
#     weights_path = "save_weights/u2net_yuanban_best.pth"
#     img_path = "1938_4.jpg"
#     threshold = 0.5
#
#     assert os.path.exists(img_path), f"image file {img_path} dose not exists."
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     data_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize(320),
#         transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                              std=(0.229, 0.224, 0.225))
#     ])
#
#     origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#
#     h, w = origin_img.shape[:2]
#     img = data_transform(origin_img)
#     img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]
#
#     model = u2net_full()
#
#     #weights = torch.load(weights_path, map_location=torch.device('cuda:0'))   #cpu
#     weights = torch.load(weights_path, map_location='cpu')
#     if "model" in weights:
#         model.load_state_dict(weights["model"])
#     else:
#         model.load_state_dict(weights)
#     model.to(device)
#     model.eval()
# #load_state_dict(state_dict)
#     with torch.no_grad():
#         # init model
#         img_height, img_width = img.shape[-2:]
#         init_img = torch.zeros((1, 3, img_height, img_width), device=device)
#         model(init_img)
#
#         t_start = time_synchronized()
#         pred = model(img)
#         t_end = time_synchronized()
#         print("inference time: {}".format(t_end - t_start))
#         pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
#
#         pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
#         pred_mask = np.where(pred > threshold, 1, 0)  #大于threshold是1，小于threshold是0   21行定义为0.5
#         origin_img = np.array(origin_img, dtype=np.uint8)
#         seg_img = origin_img * pred_mask[..., None]  #按位相乘 为1保留 为0黑色
#         plt.imshow(seg_img)
#         plt.show()
#         cv2.imwrite("pred_result.png", cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():  #改进的预测
    #weights_path = "save_weights/u2net_decoder_100epoch.pth"
    #weights_path = "save_weights/u2net_yuanban_best.pth"

    weights_path = "save_weights/DNANET.pth"
    img_path = "67_1.jpg"
    threshold = 0.5   #0.3

    assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),   #320  1280
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    h, w = origin_img.shape[:2]
    img = data_transform(origin_img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

    model = u2net_full()

    #weights = torch.load(weights_path, map_location=torch.device('cuda:0'))   #cpu
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()
#load_state_dict(state_dict)
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        pred = model(img)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        binary_mask = np.where(pred > threshold, 255, 0).astype(np.uint8)
        binary_img = binary_mask  # 只保留二值化图像
        window_name = 'Binary Segmentation Result'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w // 2, h // 2)
        cv2.imshow(window_name, binary_img)
        cv2.imwrite("binary_result.png", binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # pred_mask = np.where(pred > threshold, 1, 0)  #大于threshold是1，小于threshold是0   21行定义为0.5
        # origin_img = np.array(origin_img, dtype=np.uint8)
        # seg_img = origin_img * pred_mask[..., None]  #按位相乘 为1保留 为0黑色
        # plt.imshow(seg_img)
        # plt.show()
        # cv2.imwrite("pred_result.png", cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))



if __name__ == '__main__':
    main()
