import os
from time import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from data import *
from net import *

def binary_dice(pred, label):
    assert len(pred.shape) == len(label.shape)
    intersection = np.sum(np.logical_and(pred == 1, label == 1))
    union = np.sum(pred == 1) + np.sum(label == 1)
    dice = 2.0 * intersection / (union + 1e-10)
    return dice

def binary_sensitivity(gt, seg):
    assert len(gt.shape) == len(seg.shape)
    tp = np.sum(np.logical_and(gt > 0, seg > 0))
    tn = np.sum(np.logical_and(gt == 0, seg == 0))
    fp = np.sum(np.logical_and(gt == 0, seg > 0))
    fn = np.sum(np.logical_and(gt > 0, seg == 0))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def prediction(data_path=False, model_path=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = r'dataset/test'  # todo: 测试集路径

    outputs_path = 'outputs'  # todo 测试结果保存路径
    if os.path.exists(outputs_path) is False:
        os.mkdir(outputs_path)

    dice_list = []
    sens_list = []
    spec_list = []

    print('test')
    net = SFD_Mamba2Net().to(device)
    net.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    data_loader = torch.utils.data.DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)

    net.eval()
    st = time()

    for batch_idx, (data, label, segment_name) in enumerate(data_loader):
        with torch.no_grad():
            label = label.to(device)
            pred = net(data.to(device))
            pred = pred.squeeze().cpu().numpy()
            label = label.squeeze().cpu().numpy()
            #print("Pred shape:", pred.shape)

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            save_img = (pred * 255).astype(np.uint8)
            image = Image.fromarray(save_img)
            image.save(f'{outputs_path}/{segment_name[0]}')

            dice = binary_dice(pred, label)
            sens, spec = binary_sensitivity(label, pred)
            dice_list.append(dice)
            sens_list.append(sens)
            spec_list.append(spec)
            average_dice = np.mean(dice_list)
            average_sens = np.mean(sens_list)
            average_spec = np.mean(spec_list)

            et = time()

    print(f'val_dice:{average_dice}, val_sens:{average_sens}, val_spec:{average_spec}')
    return average_dice, average_sens, average_spec

def test_multiple_weights(weights_dir, data_path):
    # 遍历目录中的所有权重文件
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]

    if not weight_files:
        print("没有找到任何权重文件！")
        return

    best_dice = -1
    best_weight = None

    # 遍历每个权重文件
    for weight_file in weight_files:
        model_path = os.path.join(weights_dir, weight_file)
        print(f'正在测试权重文件: {model_path}')

        avg_dice, avg_sens, avg_spec = prediction(data_path=data_path, model_path=model_path)

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_weight = weight_file

        result_filename = os.path.join('test_results', f"{os.path.splitext(weight_file)[0]}_results.txt")
        if not os.path.exists('test_results'):
            os.mkdir('test_results')

        with open(result_filename, 'w') as f:
            f.write(f"Test Results for {weight_file}:\n")
            f.write(f"Average Dice: {avg_dice:.4f}\n")
            f.write(f"Average Sensitivity: {avg_sens:.4f}\n")
            f.write(f"Average Specificity: {avg_spec:.4f}\n")

        print(f"结果已保存到 {result_filename}")

    if best_weight:
        print(f"最优权重文件: {best_weight}，对应的Dice值: {best_dice:.4f}")
    else:
        print("未找到最优权重文件。")

if __name__ == '__main__':
    weights_dir = 'weight'
    data_path = 'dataset/test'
    test_multiple_weights(weights_dir, data_path)