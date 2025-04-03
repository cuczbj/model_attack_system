import torch
import torch.nn.functional as F
import kornia.augmentation as K
import time
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import random
import numpy as np

import models.losses as L
# from evaluation import get_knn_dist, calc_fid
# from models.classifiers import VGG16, IR152, FaceNet, FaceNet64
# from models.resnet64 import ResNetGenerator
from utils import save_tensor_images, tensor_to_base64,image_file_to_base64
from PIL import Image
from evaluation import test_evaluation


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

# def plot_loss_curves(loss_history, save_path='loss_curves.png'):
#     """绘制损失下降曲线"""
#     plt.figure(figsize=(12, 6))
    
#     for seed, losses in loss_history.items():
#         # 平滑处理（移动平均）
#         window_size = 50
#         smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        
#         plt.plot(smoothed_losses, label=f'Seed {seed}', alpha=0.7)
    
#     plt.xlabel('Steps')
#     plt.ylabel('Smoothed Loss')
#     plt.title('Training Loss Curves (Smoothed)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

def inversion_attack(
    args,            # 参数对象
    G,               # 生成器
    T,               # 目标模型
    target_id,       # 目标类别ID
    num_seeds=5,     # 随机种子数量
    return_images=True,
    device=device    # 设备
):
    device = next(G.parameters()).device
    iden = target_id.view(-1).long().to(device) if isinstance(target_id, torch.Tensor) \
           else torch.tensor([target_id], device=device).long()
    bs = iden.shape[0]

    # 数据增强,固定定为64*64，这里就不传参了，这个问题后面处理吧
    aug_list = K.AugmentationSequential(
        K.RandomResizedCrop((64, 64), scale=(0.8, 1.0)),
        K.ColorJitter(brightness=0.2, contrast=0.2),
        K.RandomHorizontalFlip(),
        same_on_batch=True
    )

    # 结果容器
    all_images = []
    success_images = []
    loss_history = {seed: [] for seed in range(num_seeds)}  # 记录损失

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        
        z = torch.randn(bs, args.gen_dim_z, device=device)
        z.requires_grad_(True)
        
        # 改进的优化器配置
        optimizer = torch.optim.AdamW([z], lr=args.lr, weight_decay=0.01)
        
        # 优化学习率调度器（余弦退火）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iter_times, eta_min=args.lr/10)
        
        best_loss = float('inf')
        best_z = None

        for step in range(args.iter_times):
            optimizer.zero_grad()
            
            fake = G(z, iden)
            fake_aug1 = aug_list(fake)
            fake_aug2 = aug_list(fake)
            
            logits1 = T(fake_aug1)[-1]
            logits2 = T(fake_aug2)[-1]
            
            if args.inv_loss_type == 'ce':
                loss = L.cross_entropy_loss(logits1, iden) + L.cross_entropy_loss(logits2, iden)
            elif args.inv_loss_type == 'margin':
                loss = (L.max_margin_loss(logits1, iden) + L.max_margin_loss(logits2, iden))
            elif args.inv_loss_type == 'poincare':
                loss = (L.poincare_loss(logits1, iden) + L.poincare_loss(logits2, iden))

            # 记录损失
            loss_history[seed].append(loss.item())
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            # 保存最佳z
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.clone().detach()

            if (step + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Seed {seed}, Step {step+1}: Loss = {loss.item():.4f}, LR = {current_lr:.2e}')

        # 使用最佳z生成最终图像
        with torch.no_grad():
            final_images = G(best_z, iden)
            all_images.append(final_images.cpu())
            
            preds = T(final_images)[-1].argmax(dim=1)
            for i in range(bs):
                if preds[i] == iden[i]:
                    success_images.append(final_images[i].cpu())

    # 绘制损失曲线
    # plot_loss_curves(loss_history)

    if return_images:
        return {
            'all_images': torch.cat(all_images, dim=0),
            'success_images': torch.stack(success_images, dim=0) if success_images else None,
            'loss_history': loss_history  # 返回损失历史记录
        }

def PIG_attack(target_label,  model, G,  h, w, channel, device, task_id=None):
    """
    基于PLG的攻击
    :param target_label: 目标类别
    :param model: 目标模型   输入为(1,channel,h,w，这里已经载入完毕)
    :param h: 图像高度
    :param w: 图像宽度
    :param channel: 图像通道数
    :param device: 设备
    :param task_id: 任务ID
    :return: 攻击后的图像
    """
    class Args:
        inv_loss_type = 'margin'
        lr = 2e-2
        iter_times = 1500
        gen_dim_z = 128
    args = Args()
    print("参数配置完成")
    
    results = inversion_attack(args, G, model, target_label, num_seeds=5, device=device)
    # 保存结果
    if results['success_images'] is not None:
        save_image(results['success_images'], './result/PLG_MI_Inversion/success_imgs/{}/{}_success_attacks.png'.format(target_label,target_label), nrow=5, normalize=True)
    save_image(results['all_images'], './result/PLG_MI_Inversion/all_imgs/{}/{}_all_generated.png'.format(target_label,target_label), nrow=5, normalize=True)
    
    print("攻击完成，结果已保存")

    # 测试评估的结果# 取出所有生成的图像,后面应该放到server里
    all_images_tensor = results['all_images']  # 形状: (N, C, H, W)
    accuary = test_evaluation(all_images_tensor)

    #返回 Base64 编码的图片
    image_path = f"./result/PLG_MI_Inversion/success_imgs/{target_label}/{target_label}_success_attacks.png"
    base64_img = image_file_to_base64(image_path)
    return base64_img

# 使用示例
# if __name__ == "__main__":
#     class Args:
#         inv_loss_type = 'margin'
#         lr = 2e-2
#         iter_times = 1500
#         gen_dim_z = 128
    
#     args = Args()
#     print("参数配置完成")
    
#     # 加载模型
#     G = ResNetGenerator(num_classes=1000).cuda().eval()
#     T = VGG16(n_classes=1000).cuda().eval()
#     G.load_state_dict(torch.load('./PLG_MI_Results/ffhq/VGG16/gen_latest.pth.tar')['model'])
#     G=G.to(device)
#     T.load_state_dict(torch.load('./checkpoints/target_model/VGG16_88.26.tar')['state_dict'], strict=False)
#     T=T.to(device)

#     # 执行攻击
#     results = inversion_attack(args, G, T, target_id=0, num_seeds=5)
    
#     # 保存结果
#     if results['success_images'] is not None:
#         save_image(results['success_images'], 'success_attacks.png', nrow=5, normalize=True)
#     save_image(results['all_images'], 'all_generated.png', nrow=5, normalize=True)
    
#     print("攻击完成，结果已保存")