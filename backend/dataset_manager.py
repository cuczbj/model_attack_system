# dataset_manager.py
import os
import uuid
import zipfile
import shutil
import logging
import json
import numpy as np
import time
import sqlite3
import tempfile
from datetime import datetime
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import base64

# 尝试导入图像质量评估库
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    logging.warning("未安装skimage库，将使用自定义的PSNR和SSIM函数")
    HAS_SKIMAGE = False

# 若skimage导入失败，提供替代实现
if not HAS_SKIMAGE:
    def peak_signal_noise_ratio(img1, img2, data_range=None):
        """计算PSNR (Peak Signal-to-Noise Ratio)"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        if data_range is None:
            data_range = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
        return 20 * np.log10(data_range / np.sqrt(mse))
    
    def structural_similarity(img1, img2, data_range=None, **kwargs):
        """计算SSIM (Structural Similarity Index)的简化实现"""
        if data_range is None:
            data_range = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
        
        # 计算均值
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # 计算方差和协方差
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM计算的常数
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # 计算SSIM
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = numerator / denominator
        
        return ssim

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dataset_manager')

# 数据集根目录
DATASETS_DIR = './datasets'
RESULT_DIR = './result/attack'

# 数据库连接
def get_db_connection():
    conn = sqlite3.connect('attack_system.db')
    conn.row_factory = sqlite3.Row
    return conn

# 初始化数据库
def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        description TEXT,
        upload_time TEXT NOT NULL,
        class_count INTEGER DEFAULT 0,
        image_count INTEGER DEFAULT 0,
        format TEXT,
        metadata TEXT,
        model_name TEXT
    )
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS evaluation_results (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        psnr REAL DEFAULT 0,
        ssim REAL DEFAULT 0,
        fid REAL,
        mse REAL DEFAULT 0,
        target_accuracy REAL DEFAULT 0,
        perceptual_similarity REAL,
        original_dataset TEXT,
        create_time TEXT NOT NULL,
        metrics TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks (id),
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
    ''')
    conn.commit()
    conn.close()

# 确保目录存在
def ensure_dirs():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

# 数据集上传与处理函数
def process_uploaded_dataset(dataset_file, dataset_name: str, description: str) -> Dict:
    """处理上传的数据集文件"""
    ensure_dirs()
    
    # 生成唯一ID
    dataset_id = f"dataset-{uuid.uuid4().hex[:8]}"
    dataset_path = os.path.join(DATASETS_DIR, dataset_id)
    os.makedirs(dataset_path, exist_ok=True)
    
    # 保存上传的文件
    temp_path = os.path.join(DATASETS_DIR, f"temp_{dataset_id}.zip")
    dataset_file.save(temp_path)
    
    # 解析zip文件
    try:
        # 解压文件
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # 分析数据集结构
        class_count, image_count, format_info = analyze_dataset_structure(dataset_path)
        
        # 存储数据集信息到数据库
        conn = get_db_connection()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute('''
        INSERT INTO datasets (id, name, path, description, upload_time, class_count, image_count, format, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id,
            dataset_name,
            dataset_path,
            description,
            now,
            class_count,
            image_count,
            format_info,
            json.dumps({
                "creation_date": now,
                "original_filename": dataset_file.filename
            })
        ))
        conn.commit()
        conn.close()
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "id": dataset_id,
            "name": dataset_name,
            "path": dataset_path,
            "description": description,
            "class_count": class_count,
            "image_count": image_count,
            "format": format_info,
            "upload_time": now
        }
        
    except Exception as e:
        logger.error(f"处理数据集时出错: {str(e)}")
        # 清理临时文件和目录
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        raise RuntimeError(f"处理数据集失败: {str(e)}")

def analyze_dataset_structure(dataset_path: str) -> Tuple[int, int, str]:
    """分析数据集结构，返回类别数、图像数和格式信息"""
    class_count = 0
    image_count = 0
    format_info = "unknown"
    
    # 处理常见的数据集结构
    # 1. 按类别分目录
    subdirs = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
    
    if subdirs:
        # 数据集按类别分目录
        class_count = len(subdirs)
        format_info = "class_directories"
        
        for class_dir in subdirs:
            class_path = os.path.join(dataset_path, class_dir)
            image_files = [f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            image_count += len(image_files)
    else:
        # 检查是否为扁平结构
        image_files = [f for f in os.listdir(dataset_path) 
                      if os.path.isfile(os.path.join(dataset_path, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if image_files:
            # 扁平结构，所有图像在一个目录
            image_count = len(image_files)
            class_count = 1  # 假设只有一个类别
            format_info = "flat_directory"
    
    logger.info(f"数据集分析结果: {class_count}个类别, {image_count}张图像, 格式: {format_info}")
    return class_count, image_count, format_info

def get_dataset_info(dataset_id: str) -> Dict:
    """获取数据集信息"""
    conn = get_db_connection()
    dataset = conn.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,)).fetchone()
    conn.close()
    
    if dataset is None:
        raise ValueError(f"数据集不存在: {dataset_id}")
    
    return dict(dataset)

def get_all_datasets() -> List[Dict]:
    """获取所有数据集列表"""
    conn = get_db_connection()
    datasets = conn.execute('SELECT * FROM datasets ORDER BY upload_time DESC').fetchall()
    conn.close()
    
    return [dict(dataset) for dataset in datasets]

def get_dataset_class_image(dataset_id: str, class_label: int) -> str:
    """获取指定数据集和类别的代表图像，使用目录名称作为类别标签"""
    try:
        dataset_info = get_dataset_info(dataset_id)
        dataset_path = dataset_info['path']
        str_label = str(class_label)  # 将类别标签转为字符串
        
        logger.info(f"查找数据集图像: dataset_id={dataset_id}, class_label={class_label}, 路径={dataset_path}")
        
        # 存储所有找到的图像路径
        found_images = []
        
        # 方法1: 直接搜索整个目录树，查找任何位置的匹配类别目录
        for root, dirs, files in os.walk(dataset_path):
            # 检查当前目录名是否匹配类别标签
            if os.path.basename(root) == str_label:
                # 在匹配的目录中查找图像文件
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                if image_files:
                    image_path = os.path.join(root, image_files[0])
                    logger.info(f"方法1找到图像: {image_path}")
                    found_images.append(image_path)
        
        # 方法2: 特别处理sjj/5这样的嵌套结构
        for parent_dir in os.listdir(dataset_path):
            parent_path = os.path.join(dataset_path, parent_dir)
            if os.path.isdir(parent_path):
                # 检查子目录
                for d in os.listdir(parent_path):
                    sub_dir_path = os.path.join(parent_path, d)
                    if os.path.isdir(sub_dir_path) and d == str_label:
                        # 找到匹配类别的子目录
                        image_files = [f for f in os.listdir(sub_dir_path) 
                                      if os.path.isfile(os.path.join(sub_dir_path, f)) and 
                                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                        
                        if image_files:
                            image_path = os.path.join(sub_dir_path, image_files[0])
                            logger.info(f"方法2找到图像: {image_path}")
                            found_images.append(image_path)
        
        # 如果找到了图像，返回第一个
        if found_images:
            logger.info(f"返回找到的图像: {found_images[0]}")
            return found_images[0]
        
        # 找不到时记录详细信息
        logger.error(f"无法找到类别{class_label}的图像，数据集路径: {dataset_path}")
        # 记录目录结构以便调试
        dir_structure = []
        for root, dirs, files in os.walk(dataset_path):
            rel_path = os.path.relpath(root, dataset_path)
            if rel_path != '.':
                dir_structure.append(rel_path)
        
        logger.error(f"数据集目录结构: {dir_structure}")
        raise ValueError(f"无法找到类别标签 {class_label} 对应的图像。数据集目录结构: {dir_structure}")
        
    except Exception as e:
        logger.error(f"获取数据集类别图像时出错: {str(e)}")
        raise RuntimeError(f"获取类别图像失败: {str(e)}")
# 添加到dataset_manager.py
def process_uploaded_dataset_for_model(dataset_file, dataset_name: str, model_name: str, description: str) -> Dict:
    """处理上传的与模型关联的数据集文件"""
    ensure_dirs()
    
    # 生成唯一ID，但使用模型名称作为前缀，确保关联性
    dataset_id = f"{model_name}-dataset-{uuid.uuid4().hex[:8]}"
    dataset_path = os.path.join(DATASETS_DIR, dataset_id)
    os.makedirs(dataset_path, exist_ok=True)
    
    # 保存上传的文件
    temp_path = os.path.join(DATASETS_DIR, f"temp_{dataset_id}.zip")
    dataset_file.save(temp_path)
    
    # 解析zip文件
    try:
        # 解压文件
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # 分析数据集结构
        class_count, image_count, format_info = analyze_dataset_structure(dataset_path)
        
        # 存储数据集信息到数据库
        conn = get_db_connection()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute('''
        INSERT INTO datasets (id, name, path, description, upload_time, class_count, image_count, format, metadata, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id,
            dataset_name,
            dataset_path,
            description,
            now,
            class_count,
            image_count,
            format_info,
            json.dumps({
                "creation_date": now,
                "original_filename": dataset_file.filename,
                "model_name": model_name
            }),
            model_name
        ))
        conn.commit()
        conn.close()
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "id": dataset_id,
            "name": dataset_name,
            "path": dataset_path,
            "description": description,
            "class_count": class_count,
            "image_count": image_count,
            "format": format_info,
            "upload_time": now,
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"处理模型数据集时出错: {str(e)}")
        # 清理临时文件和目录
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        raise RuntimeError(f"处理模型数据集失败: {str(e)}")

def get_dataset_for_model(model_name: str) -> Dict:
    """获取指定模型对应的数据集"""
    conn = get_db_connection()
    dataset = conn.execute('SELECT * FROM datasets WHERE model_name = ?', (model_name,)).fetchone()
    conn.close()
    
    if dataset is None:
        raise ValueError(f"未找到模型 {model_name} 对应的数据集")
    
    return dict(dataset)


# 评估相关功能
def evaluate_attack_result(task_id: str, dataset_id: str, metrics: List[str]) -> Dict:
    """评估攻击结果的效果"""
    try:
        # 获取任务信息
        conn = get_db_connection()
        task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
        
        if task is None:
            raise ValueError(f"任务不存在: {task_id}")
        
        task_dict = dict(task)
        target_label = task_dict.get('target_label')
        model_name = task_dict.get('model')
        
        # 获取攻击生成的图像
        attack_image_path = os.path.join(RESULT_DIR, f"{task_id}_{target_label}.png")
        
        # 尝试其他可能的路径
        if not os.path.exists(attack_image_path):
            attack_image_path = os.path.join(RESULT_DIR, f"inverted_{target_label}.png")
        
        # 再尝试一种可能的路径 (PIG攻击)
        if not os.path.exists(attack_image_path):
            attack_image_path = os.path.join("./result/PLG_MI_Inversion/success_imgs", 
                                          f"{target_label}/0_attack_iden_{target_label}_0.png")
        
        # 再尝试一种可能的路径
        if not os.path.exists(attack_image_path):
            attack_image_path = os.path.join("./result/PLG_MI_Inversion/all_imgs", 
                                          f"{target_label}/attack_iden_{target_label}_0.png")
        
        if not os.path.exists(attack_image_path):
            raise ValueError(f"攻击结果图像不存在: {attack_image_path}")
        
        # 获取模型对应的数据集
        conn = get_db_connection()
        dataset = conn.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,)).fetchone()
        
        if dataset is None:
            raise ValueError(f"未找到模型 {model_name} 对应的数据集")
        
        dataset_dict = dict(dataset)
        dataset_id = dataset_dict['id']
        
        # 获取原始数据集中的对应图像
        original_image_path = get_dataset_class_image(dataset_id, target_label)
        
        # 加载图像
        attack_image = np.array(Image.open(attack_image_path).convert('L'))
        original_image = np.array(Image.open(original_image_path).convert('L'))
        
        # 确保图像尺寸一致
        if attack_image.shape != original_image.shape:
            # 调整攻击图像大小以匹配原始图像
            attack_image_pil = Image.fromarray(attack_image)
            attack_image_pil = attack_image_pil.resize(
                (original_image.shape[1], original_image.shape[0]), 
                Image.LANCZOS
            )
            attack_image = np.array(attack_image_pil)
        
        # 计算评估指标
        evaluation_results = {}
        
        # 计算PSNR（信噪比）
        if 'psnr' in metrics:
            psnr_value = peak_signal_noise_ratio(original_image, attack_image, data_range=255)
            evaluation_results['psnr'] = float(psnr_value)
        
        # 计算SSIM（结构相似性）
        if 'ssim' in metrics:
            ssim_value = structural_similarity(original_image, attack_image, data_range=255)
            evaluation_results['ssim'] = float(ssim_value)
        
        # 计算MSE（均方误差）
        if 'mse' in metrics:
            mse_value = np.mean((original_image - attack_image) ** 2)
            evaluation_results['mse'] = float(mse_value)
        
        # 目标准确率（此处需要模型预测）
        if 'target_accuracy' in metrics:
            # 使用实际模型预测
            from predict import predict_target_model
            predicted_class, _ = predict_target_model(
                Image.fromarray(attack_image), 
                task_dict['model'],
                original_image.shape[0],
                original_image.shape[1],
                int(dataset_dict['class_count'])
            )
            evaluation_results['target_accuracy'] = 1.0 if predicted_class == target_label else 0.0
        
        # FID评分（如果需要）
        if 'fid' in metrics:
            # 实际实现可能需要调用其他库计算FID
            pass
        
        # 保存评估结果
        evaluation_id = f"eval-{uuid.uuid4().hex[:8]}"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 存储到数据库
        conn.execute('''
        INSERT INTO evaluation_results (
            id, task_id, dataset_id, psnr, ssim, fid, mse, target_accuracy, 
            original_dataset, create_time, metrics
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evaluation_id,
            task_id,
            dataset_id,
            evaluation_results.get('psnr', 0),
            evaluation_results.get('ssim', 0),
            evaluation_results.get('fid'),
            evaluation_results.get('mse', 0),
            evaluation_results.get('target_accuracy', 0),
            dataset_id,
            now,
            json.dumps(evaluation_results)
        ))
        conn.commit()
        conn.close()
        
        # 返回评估结果
        evaluation_results['id'] = evaluation_id
        evaluation_results['task_id'] = task_id
        evaluation_results['create_time'] = now
        
        return evaluation_results
    
    except Exception as e:
        logger.error(f"评估攻击结果时出错: {str(e)}")
        raise RuntimeError(f"评估失败: {str(e)}")
    
def get_evaluation_results(task_id: str = None) -> List[Dict]:
    """获取评估结果，可以按任务ID过滤"""
    conn = get_db_connection()
    
    if task_id:
        results = conn.execute(
            'SELECT * FROM evaluation_results WHERE task_id = ? ORDER BY create_time DESC', 
            (task_id,)
        ).fetchall()
    else:
        results = conn.execute(
            'SELECT * FROM evaluation_results ORDER BY create_time DESC'
        ).fetchall()
    
    conn.close()
    
    return [dict(result) for result in results]

def generate_comparison_image(task_id: str, dataset_id: str) -> str:
    """生成攻击图像与原始图像的对比图"""
    try:
        # 获取任务信息
        conn = get_db_connection()
        task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
        
        if task is None:
            raise ValueError(f"任务不存在: {task_id}")
        
        task_dict = dict(task)
        target_label = task_dict.get('target_label')
        
        # 获取攻击生成的图像
        attack_image_path = os.path.join(RESULT_DIR, f"{task_id}_{target_label}.png")
        
        # 尝试其他可能的路径
        if not os.path.exists(attack_image_path):
            attack_image_path = os.path.join(RESULT_DIR, f"inverted_{target_label}.png")
        
        # 再尝试一种可能的路径 (PIG攻击)
        if not os.path.exists(attack_image_path):
            attack_image_path = os.path.join("./result/PLG_MI_Inversion/success_imgs", 
                                           f"{target_label}/0_attack_iden_{target_label}_0.png")
        
        # 再尝试一种可能的路径
        if not os.path.exists(attack_image_path):
            attack_image_path = os.path.join("./result/PLG_MI_Inversion/all_imgs", 
                                           f"{target_label}/attack_iden_{target_label}_0.png")
        
        if not os.path.exists(attack_image_path):
            raise ValueError(f"攻击结果图像不存在: {attack_image_path}")
        
        # 获取原始数据集中的对应图像
        original_image_path = get_dataset_class_image(dataset_id, target_label)
        
        # 加载图像
        attack_image = Image.open(attack_image_path).convert('L')
        original_image = Image.open(original_image_path).convert('L')
        
        # 确保图像尺寸一致
        if attack_image.size != original_image.size:
            # 调整攻击图像大小以匹配原始图像
            attack_image = attack_image.resize(original_image.size, Image.LANCZOS)
        
        # 创建matplotlib图表
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 显示原始图像
        axes[0].imshow(np.array(original_image), cmap='gray')
        axes[0].set_title('原始隐私图像')
        axes[0].axis('off')
        
        # 显示攻击结果图像
        axes[1].imshow(np.array(attack_image), cmap='gray')
        axes[1].set_title('攻击重建图像')
        axes[1].axis('off')
        
        # 添加评估结果
        conn = get_db_connection()
        evaluation = conn.execute(
            'SELECT * FROM evaluation_results WHERE task_id = ? ORDER BY create_time DESC LIMIT 1', 
            (task_id,)
        ).fetchone()
        conn.close()
        
        if evaluation:
            eval_dict = dict(evaluation)
            fig.suptitle(f"图像对比 - PSNR: {eval_dict['psnr']:.2f}dB, SSIM: {eval_dict['ssim']:.4f}", fontsize=14)
        
        # 保存图表到内存中的图像
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # 关闭图表
        plt.close(fig)
        
        # 转换为base64字符串
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64
    
    except Exception as e:
        logger.error(f"生成对比图时出错: {str(e)}")
        raise RuntimeError(f"生成对比图失败: {str(e)}")

def generate_evaluation_charts() -> Dict[str, str]:
    """生成评估结果图表"""
    try:
        # 获取所有评估结果
        conn = get_db_connection()
        results = conn.execute('''
            SELECT er.*, t.attack_type, t.model 
            FROM evaluation_results er
            JOIN tasks t ON er.task_id = t.id
            ORDER BY er.create_time DESC
        ''').fetchall()
        conn.close()
        
        if not results:
            raise ValueError("没有可用的评估结果")
        
        charts = {}
        
        # 按攻击方法分组的PSNR对比图
        psnr_by_attack = {}
        for r in results:
            result = dict(r)
            attack_type = result.get('attack_type', 'unknown')
            if attack_type not in psnr_by_attack:
                psnr_by_attack[attack_type] = []
            psnr_by_attack[attack_type].append(result.get('psnr', 0))
        
        if psnr_by_attack:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            positions = range(len(psnr_by_attack))
            
            # 对于每种攻击方法，绘制箱线图
            ax.boxplot([values for values in psnr_by_attack.values()])
            ax.set_xticklabels(list(psnr_by_attack.keys()))
            ax.set_title('各攻击方法PSNR对比')
            ax.set_ylabel('PSNR (dB)')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            charts['psnr_comparison'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
        
        # 按攻击方法分组的SSIM对比图
        ssim_by_attack = {}
        for r in results:
            result = dict(r)
            attack_type = result.get('attack_type', 'unknown')
            if attack_type not in ssim_by_attack:
                ssim_by_attack[attack_type] = []
            ssim_by_attack[attack_type].append(result.get('ssim', 0))
        
        if ssim_by_attack:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            positions = range(len(ssim_by_attack))
            
            # 对于每种攻击方法，绘制箱线图
            ax.boxplot([values for values in ssim_by_attack.values()])
            ax.set_xticklabels(list(ssim_by_attack.keys()))
            ax.set_title('各攻击方法SSIM对比')
            ax.set_ylabel('SSIM')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            charts['ssim_comparison'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
        
        return charts
    
    except Exception as e:
        logger.error(f"生成评估图表时出错: {str(e)}")
        raise RuntimeError(f"生成评估图表失败: {str(e)}")

# 批量评估功能
def run_batch_evaluation(config: Dict) -> str:
    """运行批量评估任务，返回评估ID"""
    try:
        dataset_id = config.get('dataset')
        attack_methods = config.get('attackMethods', [])
        label_range = config.get('labelRange', {'start': 0, 'end': 9})
        samples_per_label = config.get('samplesPerLabel', 5)
        evaluate_all = config.get('evaluateAll', False)
        
        # 获取数据集信息
        dataset_info = get_dataset_info(dataset_id)
        
        # 生成批量评估ID
        evaluation_id = f"batch-{uuid.uuid4().hex[:8]}"
        
        # 存储批量评估任务
        conn = get_db_connection()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取需要评估的攻击任务
        tasks_query = 'SELECT * FROM tasks WHERE status = ?'
        tasks_params = ('completed',)
        
        if not evaluate_all:
            tasks_query += ' AND target_label BETWEEN ? AND ?'
            tasks_params = tasks_params + (label_range['start'], label_range['end'])
        
        if attack_methods:
            placeholders = ','.join(['?'] * len(attack_methods))
            tasks_query += f' AND attack_type IN ({placeholders})'
            tasks_params = tasks_params + tuple(attack_methods)
        
        tasks = conn.execute(tasks_query, tasks_params).fetchall()
        
        # 计算总样本数
        total_samples = len(tasks)
        
        # 存储批量评估信息
        conn.execute('''
        INSERT INTO evaluations (id, name, dataset, target_model, attack_methods, status, progress, 
                                total_samples, completed_samples, start_time, parameters, create_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evaluation_id,
            f"批量评估-{evaluation_id}",
            dataset_id,
            "多个模型",
            json.dumps(attack_methods),
            'running',
            0,
            total_samples,
            0,
            now,
            json.dumps(config),
            now
        ))
        conn.commit()
        
        # 启动批量评估进程
        import threading
        evaluation_thread = threading.Thread(
            target=_execute_batch_evaluation,
            args=(evaluation_id, tasks, dataset_id, config)
        )
        evaluation_thread.daemon = True
        evaluation_thread.start()
        
        return evaluation_id
    
    except Exception as e:
        logger.error(f"创建批量评估任务时出错: {str(e)}")
        raise RuntimeError(f"创建批量评估失败: {str(e)}")

def _execute_batch_evaluation(evaluation_id: str, tasks: List, dataset_id: str, config: Dict):
    """执行批量评估（在单独线程中运行）"""
    try:
        total_samples = len(tasks)
        completed_samples = 0
        
        # 按攻击方法分组的结果
        results_by_method = {}
        
        # 处理每个任务
        for task in tasks:
            task_dict = dict(task)
            task_id = task_dict['id']
            attack_type = task_dict['attack_type']
            
            # 检查评估任务是否被停止
            conn = get_db_connection()
            status = conn.execute('SELECT status FROM evaluations WHERE id = ?', 
                                (evaluation_id,)).fetchone()[0]
            conn.close()
            
            if status == 'stopped':
                logger.info(f"批量评估任务 {evaluation_id} 已被停止")
                return
            
            # 评估当前任务
            try:
                result = evaluate_attack_result(
                    task_id, 
                    dataset_id, 
                    ['psnr', 'ssim', 'mse', 'target_accuracy', 'fid']
                )
                
                # 按攻击方法分组
                if attack_type not in results_by_method:
                    results_by_method[attack_type] = {
                        'method': attack_type,
                        'psnr_values': [],
                        'ssim_values': [],
                        'mse_values': [],
                        'accuracy_values': [],
                        'fid_values': []
                    }
                
                # 添加结果
                method_results = results_by_method[attack_type]
                method_results['psnr_values'].append(result.get('psnr', 0))
                method_results['ssim_values'].append(result.get('ssim', 0))
                method_results['mse_values'].append(result.get('mse', 0))
                method_results['accuracy_values'].append(result.get('target_accuracy', 0))
                
                if 'fid' in result and result['fid'] is not None:
                    method_results['fid_values'].append(result['fid'])
            
            except Exception as e:
                logger.error(f"评估任务 {task_id} 时出错: {e}")
            
            # 更新进度
            completed_samples += 1
            progress = int(completed_samples * 100 / total_samples)
            
            conn = get_db_connection()
            conn.execute('''
            UPDATE evaluations SET 
                completed_samples = ?, 
                progress = ?
            WHERE id = ?
            ''', (
                completed_samples,
                progress,
                evaluation_id
            ))
            conn.commit()
            conn.close()
        
        # 计算每种攻击方法的统计结果
        aggregated_results = []
        
        for method, data in results_by_method.items():
            psnr_values = data['psnr_values']
            ssim_values = data['ssim_values']
            mse_values = data['mse_values']
            accuracy_values = data['accuracy_values']
            fid_values = data['fid_values']
            
            # 计算平均值
            avg_result = {
                'method': method,
                'accuracy': sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0,
                'successRate': sum(1 for v in accuracy_values if v > 0.5) / len(accuracy_values) if accuracy_values else 0,
                'psnr': sum(psnr_values) / len(psnr_values) if psnr_values else 0,
                'ssim': sum(ssim_values) / len(ssim_values) if ssim_values else 0,
                'mse': sum(mse_values) / len(mse_values) if mse_values else 0,
                'avgConfidence': 0.75,  # 示例值，实际应从模型获取
                'executionTime': 0,
                'sampleCount': len(psnr_values)
            }
            
            if fid_values:
                avg_result['fid'] = sum(fid_values) / len(fid_values)
            
            aggregated_results.append(avg_result)
        
        # 更新评估任务状态和结果
        conn = get_db_connection()
        conn.execute('''
        UPDATE evaluations SET 
            status = ?,
            end_time = ?,
            progress = 100,
            results = ?
        WHERE id = ?
        ''', (
            'completed',
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            json.dumps(aggregated_results),
            evaluation_id
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"批量评估任务 {evaluation_id} 已完成")
    
    except Exception as e:
        logger.error(f"执行批量评估时出错: {str(e)}")
        
        # 更新评估任务状态为失败
        try:
            conn = get_db_connection()
            conn.execute('''
            UPDATE evaluations SET 
                status = ?,
                end_time = ?,
                parameters = ?
            WHERE id = ?
            ''', (
                'failed',
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                json.dumps({**config, 'error': str(e)}),
                evaluation_id
            ))
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.error(f"更新评估失败状态时出错: {db_error}")

# 初始化
def init():
    """初始化模块"""
    ensure_dirs()
    init_db()
    logger.info("数据集管理模块初始化完成")

# 调用初始化
init()