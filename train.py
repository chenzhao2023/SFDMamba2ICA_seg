import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from net import *
from data import PairedDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weight_path = "weights"
data_path_s = r"dataset/source"
data_path_t = r"dataset/target"

epochs = 200
batch_size = 1
lr = 0.001
lambda_sda = 0.03

alpha, beta = 0.7, 0.3
bce_loss_seg = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()


def segmentation_loss(pred, mask, alpha=alpha, beta=beta):
    loss_bce = bce_loss_seg(pred, mask)
    loss_mse = mse_loss(torch.sigmoid(pred), mask)
    return alpha * loss_bce + beta * loss_mse


bce_loss = nn.BCEWithLogitsLoss()


def dice_coefficient(pred, target, smooth=1e-10):
    pred = (pred >= 0.5).float()
    pred, target = pred.view(-1), target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def save_checkpoint(epoch, model, optimizer):
    os.makedirs(weight_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f'{weight_path}/epoch_{epoch}.pth')


def multi_stage_lr(epoch):
    if epoch < 10:
        return 1.0
    elif epoch < 20:
        return 0.5
    elif epoch < 30:
        return 0.25
    else:
        return 0.3125


paired_dataset = PairedDataset(
    source_img_dir=os.path.join(data_path_s, "images"),
    source_label_dir=os.path.join(data_path_s, "masks"),
    target_img_dir=os.path.join(data_path_t, "images"),
    target_label_dir=os.path.join(data_path_t, "masks")
)

loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

net = SFD_Mamba2Net().to(device)

opt = optim.Adam(
    net.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=multi_stage_lr)

checkpoint = False
chek = ''
if checkpoint and chek:
    net.load_state_dict(torch.load(f'{weight_path}/{chek}.pth')['model_state_dict'])
    print(f'Successfully loaded weight: {chek}')

for epoch in range(1, epochs + 1):
    net.train()
    total_dice_s, total_dice_t = 0., 0.
    total_loss_s, total_loss_t, total_loss_sda, batches = 0., 0., 0., 0

    current_lambda_sda = lambda_sda

    for i, (img_s, mask_s, _, img_t, mask_t, _) in enumerate(loader):
        img_s, mask_s = img_s.to(device), mask_s.to(device)
        img_t, mask_t = img_t.to(device), mask_t.to(device)
        B = img_s.size(0)

        img = torch.cat([img_s, img_t], dim=0)
        mask = torch.cat([mask_s, mask_t], dim=0)

        pred, dom_out = net(img, lambd=current_lambda_sda, cond=mask)

        pred_s, pred_t = pred[:B], pred[B:]
        mask_s, mask_t = mask[:B], mask[B:]

        loss_s = segmentation_loss(pred_s, mask_s)
        loss_t = segmentation_loss(pred_t, mask_t)

        label_s = torch.zeros(B, 1, device=device)  # source=0
        label_t = torch.ones(B, 1, device=device)  # target=1
        labels = torch.cat([label_s, label_t], dim=0)
        loss_sda = bce_loss(dom_out, labels)

        loss = loss_s + loss_t + current_lambda_sda * loss_sda

        opt.zero_grad()
        loss.backward()
        opt.step()

        dice_s = dice_coefficient(torch.sigmoid(pred_s), mask_s).item()
        dice_t = dice_coefficient(torch.sigmoid(pred_t), mask_t).item()

        total_dice_s += dice_s
        total_dice_t += dice_t
        total_loss_s += loss_s.item()
        total_loss_t += loss_t.item()
        total_loss_sda += loss_sda.item()
        batches += 1

        if i % 50 == 0:
            print(f'Epoch {epoch} | Iter {i} | '
                  f'LossS={loss_s.item():.4f} LossT={loss_t.item():.4f} LossSDA={loss_sda.item():.4f} '
                  f'| DiceS={dice_s:.4f} DiceT={dice_t:.4f} | LambdaSDA={current_lambda_sda:.3f}')

    print(f'Epoch {epoch} Summary | '
          f'AvgLossS={total_loss_s / batches:.4f} AvgLossT={total_loss_t / batches:.4f} AvgLossSDA={total_loss_sda / batches:.4f} '
          f'| AvgDiceS={total_dice_s / batches:.4f} AvgDiceT={total_dice_t / batches:.4f} | LambdaSDA={current_lambda_sda:.3f}')

    if epoch % 5 == 0:
        save_checkpoint(epoch, net, opt)

    scheduler.step()
