import logging
import numpy as np
import os
import random
import statistics
import time
import torch
from argparse import ArgumentParser
from kornia import augmentation
import sys
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models.losses as L
import utils
from utils import save_tensor_images,tensor_to_base64
from models.evaluation import get_knn_dist, calc_fid
from models.classifiers import VGG16, IR152, FaceNet, FaceNet64
from models.resnet64 import ResNetGenerator

#设定随机种子，保证可复现性。
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# logger配置日志系统，输出日志信息。
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

#逆向攻击出对应标签的图像
def inversion(G, T, E, iden, itr, lr=2e-2, iter_times=1500, num_seeds=5,save_dir="",gen_dim_z=64,gen_distribution='',inv_loss_type=''):
    """输入：
    G：GAN 生成器（ResNetGenerator）
    T：目标分类器（VGG16/IR152/FaceNet64）
    E：评估模型（FaceNet）
    iden：目标身份编号"""
    save_img_dir = os.path.join(save_dir, 'all_imgs')
    success_dir = os.path.join(save_dir, 'success_imgs')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)

    bs = iden.shape[0]
    iden = iden.view(-1).long().cuda()

    G.eval()
    T.eval()
    E.eval()

    flag = torch.zeros(bs)
    no = torch.zeros(bs)  # index for saving all success attack images

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, 5))

    aug_list = augmentation.container.ImageSequential(
        augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        augmentation.ColorJitter(brightness=0.2, contrast=0.2),
        augmentation.RandomHorizontalFlip(),
        augmentation.RandomRotation(5),
    )

    """执行攻击：
    每轮攻击 60 (bs)个类别，共 5 轮
    计算Top-1 和 Top-5 攻击成功率
    计算 KNN 评估距离
    计算 FID 评估生成图像的质量"""
    for random_seed in range(num_seeds):
        tf = time.time()
        r_idx = random_seed

        set_random_seed(random_seed)

        z = utils.sample_z(
            bs, gen_dim_z, device, gen_distribution
        )
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(iter_times):

            fake = G(z, iden)

            out1 = T(aug_list(fake))[-1]
            out2 = T(aug_list(fake))[-1]

            if z.grad is not None:
                z.grad.data.zero_()

            if inv_loss_type == 'ce':
                inv_loss = L.cross_entropy_loss(out1, iden) + L.cross_entropy_loss(out2, iden)
            elif inv_loss_type == 'margin':
                inv_loss = L.max_margin_loss(out1, iden) + L.max_margin_loss(out2, iden)
            elif inv_loss_type == 'poincare':
                inv_loss = L.poincare_loss(out1, iden) + L.poincare_loss(out2, iden)

            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

            inv_loss_val = inv_loss.item()

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    fake_img = G(z, iden)
                    eval_prob = E(augmentation.Resize((112, 112))(fake_img))[-1]
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))

        with torch.no_grad():
            fake = G(z, iden)
            score = T(fake)[-1]
            eval_prob = E(augmentation.Resize((112, 112))(fake))[-1]
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

            cnt, cnt5 = 0, 0
            for i in range(bs):
                gt = iden[i].item()
                sample = G(z, iden)[i]
                all_img_class_path = os.path.join(save_img_dir, str(gt))
                if not os.path.exists(all_img_class_path):
                    os.makedirs(all_img_class_path)
                save_tensor_images(sample.detach(),
                                   os.path.join(all_img_class_path, "attack_iden_{}_{}.png".format(gt, r_idx)))
                
                if(i==bs-1):
                    base64_img = tensor_to_base64(sample.detach())

                if eval_iden[i].item() == gt:
                    seed_acc[i, r_idx] = 1
                    cnt += 1
                    flag[i] = 1
                    best_img = G(z, iden)[i]
                    success_img_class_path = os.path.join(success_dir, str(gt))
                    if not os.path.exists(success_img_class_path):
                        os.makedirs(success_img_class_path)
                    save_tensor_images(best_img.detach(), os.path.join(success_img_class_path,
                                                                       "{}_attack_iden_{}_{}.png".format(itr, gt,
                                                                                                         int(no[i]))))
                    no[i] += 1
                _, top5_idx = torch.topk(eval_prob[i], 5)
                if gt in top5_idx:
                    cnt5 += 1

            interval = time.time() - tf
            print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
            res.append(cnt * 1.0 / bs)
            res5.append(cnt5 * 1.0 / bs)
            torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.variance(res)
    acc_var5 = statistics.variance(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return base64_img,acc, acc_5, acc_var, acc_var5

#发起PIG逆向攻击
def PIG_attack(target_labels=None,model='VGG16', inv_loss_type='margin', lr=0.1, iter_times=600,
                          gen_num_features=64, gen_dim_z=128, gen_bottom_width=4,
                          gen_distribution='normal', save_dir='./result/PLG_MI_Inversion', path_G='./upload/PIG/gen_VGG16_celeba.pth.tar'):
    # Load Generator
    G = ResNetGenerator(gen_num_features, gen_dim_z, gen_bottom_width, num_classes=1000, distribution=gen_distribution)
    gen_ckpt = torch.load(path_G)['model']
    G.load_state_dict(gen_ckpt)
    G = G.cuda()

    # Load Target Model
    if model.startswith("VGG16"):
        T = VGG16(1000)
        path_T = './upload/target_model/VGG16_88.26.tar'
    elif model.startswith('IR152'):
        T = IR152(1000)
        path_T = './upload/target_model/IR152_91.16.tar'
    elif model == "FaceNet64":
        T = FaceNet64(1000)
        path_T = './upload/target_model/FaceNet64_88.50.tar'
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    # Load Evaluation Model
    E = FaceNet(1000)
    E = torch.nn.DataParallel(E).cuda()
    path_E = './upload/evaluate_model/FaceNet_95.88.tar'
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)


    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    for i in range(1):
        iden = torch.tensor([target_labels])  # 这里只攻击类别 5
        # for idx in range(5):记得这里改了之后下面的/5的要改为/3，算了，我还是改为一个变量存储吧
        batch_num =3
        for idx in range(batch_num):
            print(f"--------------------- Attack batch [{idx}]------------------------------")
            base64_img,acc, acc5, var, var5 = inversion(G, T, E, iden, itr=i, lr=lr, iter_times=iter_times,
                                             num_seeds=5,save_dir=save_dir,gen_dim_z=gen_dim_z,gen_distribution=gen_distribution,inv_loss_type=inv_loss_type)
            # iden += 60
            aver_acc += acc / batch_num
            aver_acc5 += acc5 / batch_num
            aver_var += var / batch_num
            aver_var5 += var5 / batch_num
    
    print(f"Average Acc:{aver_acc:.2f}\tAverage Acc5:{aver_acc5:.2f}\tAverage Acc_var:{aver_var:.4f}\tAverage Acc_var5:{aver_var5:.4f}")
    return base64_img

#test
# if __name__ == "__main__":
#     PIG_attack(5,"VGG16","margin")