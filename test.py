import os
import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from data import TargetDataset
from net import *

def dice_coeff(pred, label):
    pred = pred.astype(np.bool_)
    label = label.astype(np.bool_)
    intersection = np.logical_and(pred, label).sum()
    return (2. * intersection) / (pred.sum() + label.sum() + 1e-10)

def sensitivity(gt, seg):
    tp = np.logical_and(gt == 1, seg == 1).sum()
    fn = np.logical_and(gt == 1, seg == 0).sum()
    return tp / (tp + fn + 1e-10)

def specificity(gt, seg):
    tn = np.logical_and(gt == 0, seg == 0).sum()
    fp = np.logical_and(gt == 0, seg == 1).sum()
    return tn / (tn + fp + 1e-10)

def evaluate_model(weight_path, loader, device):
    print(f"\nEvaluating weight file: {weight_path}")
    net = SFD_Mamba2Net().to(device)

    checkpoint = torch.load(weight_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    net.load_state_dict(state_dict, strict=False)

    net.eval()
    dice_list, sens_list, spec_list = [], [], []
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    start_time = time.time()
    with torch.no_grad():
        for img, mask, name in loader:
            img, mask = img.to(device), mask.to(device)
            pred, _ = net(img)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            mask = mask.squeeze().cpu().numpy()

            pred_bin = (pred >= 0.5).astype(np.uint8)
            save_img = (pred_bin * 255).astype(np.uint8)
            Image.fromarray(save_img).save(os.path.join(outputs_dir, name[0]))

            dice_list.append(dice_coeff(pred_bin, mask))
            sens_list.append(sensitivity(mask, pred_bin))
            spec_list.append(specificity(mask, pred_bin))

    avg_dice = float(np.mean(dice_list))
    avg_sens = float(np.mean(sens_list))
    avg_spec = float(np.mean(spec_list))
    elapsed = time.time() - start_time

    print(f"val_dice: {avg_dice:.4f}, val_sens: {avg_sens:.4f}, val_spec: {avg_spec:.4f}, time: {elapsed:.2f}s")
    return avg_dice, avg_sens, avg_spec

def test_all_weights():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_dir = "weights"
    data_path = "dataset/test"
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]

    if not weight_files:
        print("No weight files found!")
        return

    test_results_dir = "test_results"
    os.makedirs(test_results_dir, exist_ok=True)

    dataset = TargetDataset(os.path.join(data_path, "images"),
                            os.path.join(data_path, "masks"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    best_dice, best_weight = -1, None

    for weight_file in weight_files:
        weight_path = os.path.join(weights_dir, weight_file)
        avg_dice, avg_sens, avg_spec = evaluate_model(weight_path, loader, device)

        result_path = os.path.join(test_results_dir, f"{os.path.splitext(weight_file)[0]}_results.txt")
        with open(result_path, "w") as f:
            f.write(f"Test Results for {weight_file}:\n")
            f.write(f"Average Dice: {avg_dice:.4f}\n")
            f.write(f"Average Sensitivity: {avg_sens:.4f}\n")
            f.write(f"Average Specificity: {avg_spec:.4f}\n")
        print(f"Results saved to {result_path}")

        if avg_dice > best_dice:
            best_dice, best_weight = avg_dice, weight_file

    if best_weight:
        print("\n================= Best Result =================")
        print(f"Best weight file: {best_weight}, Dice: {best_dice:.4f}")
    else:
        print("No best weight file found.")

# ----------------- Main Entry -----------------
if __name__ == "__main__":
    test_all_weights()
