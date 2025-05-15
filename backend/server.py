# server.py
import logging
import os
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from predict import train_target_model, predict_target_model
from reconstruct import reconstruct
from PIL import Image
import psutil
import json
import time
from datetime import datetime
import uuid
import sqlite3
import redis
import base64
import io
import numpy as np
from threading import Thread
import sys
import torch
from upload_importlib import get_available_models, get_available_params, load_model, image_to_tensor, MODEL_DIR, CHECKPOINT_DIR,load_G
from typing import Any, Tuple
import dataset_manager
import model_scanner
from model_scanner import get_model_scanner, ModelConfig
import torch
import json
from flask import jsonify, request
import uuid
import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 配置根日志记录器
log_file = os.path.join(log_dir, 'server.log')

# 创建日志处理器 - 使用RotatingFileHandler自动管理文件大小
handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,          # 保留5个备份文件
    encoding='utf-8'
)

# 设置日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
)
handler.setFormatter(formatter)

# 配置日志级别
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # 设置为DEBUG可以捕获所有级别的日志
root_logger.addHandler(handler)

# 保留控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# 配置特定模块的日志级别
logging.getLogger('werkzeug').setLevel(logging.INFO)  # Flask请求日志
logging.getLogger('PIL').setLevel(logging.INFO)       # 图像处理库

# 应用启动日志
logging.info("模型攻击系统服务启动中...")


# 将attack目录添加到系统路径
attack_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attack')
sys.path.append(attack_dir)
# 配置日志
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置数据库
def get_db_connection():
    conn = sqlite3.connect('attack_system.db')
    conn.row_factory = sqlite3.Row
    return conn

# 配置Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 初始化数据库
def init_db():
    conn = get_db_connection()
    # 原有的攻击任务表
    conn.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        model TEXT NOT NULL,
        attack_type TEXT NOT NULL,
        status TEXT NOT NULL,
        progress INTEGER DEFAULT 0,
        create_time TEXT NOT NULL,
        start_time TEXT,
        end_time TEXT,
        error_message TEXT,
        description TEXT,
        target_label INTEGER,
        parameters TEXT,
        image_path TEXT
    )
    ''')
    
    # 新增的评估任务表
    conn.execute('''
    CREATE TABLE IF NOT EXISTS evaluations (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        dataset TEXT NOT NULL,
        target_model TEXT NOT NULL,
        attack_methods TEXT NOT NULL,
        status TEXT NOT NULL,
        progress INTEGER DEFAULT 0,
        total_samples INTEGER NOT NULL,
        completed_samples INTEGER DEFAULT 0,
        start_time TEXT,
        end_time TEXT,
        parameters TEXT,
        results TEXT,
        create_time TEXT NOT NULL
    )
    ''')

    # 新增用户表
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        create_time TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()



# 创建Flask应用
app = Flask(__name__, static_url_path="/static", static_folder="./")
CORS(app)  # 启用CORS，允许所有源访问

app.config['PYTHON_FOLDER'] = os.path.join(os.getcwd(), './models/classifiers')
app.config['CTP_FOLDER'] = os.path.join(os.getcwd(), './checkpoint/target_model')
# app.config['ALLOWED_EXTENSIONS'] = {'pth', 'pkl','tar'}
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'tar', 'pkl', 'pth', 'py'}
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# 在启动应用时初始化数据库
init_db()

# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 简单示例，真实环境请加密密码和数据库支持
users = {
    "admin": "admin",
    "user": "admin"
}

#################################################
# 原有API端点
#################################################

# 训练目标模型（备用）
@app.route("/train", methods=["POST"])
def train():
    response = train_target_model()
    return jsonify({"message": response})

# 用户登录
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username in users and users[username] == password:
        # 登录成功，返回简单 token（演示用，生产建议用 JWT）
        token = f"token-for-{username}"
        return jsonify({"token": token}), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

# 获取任务图像
@app.route("/api/tasks/<task_id>/image", methods=["GET"])
def get_task_image(task_id):
    """获取指定任务的攻击结果图像"""
    logging.debug(f"获取任务图像: {task_id}")
    
    conn = get_db_connection()
    task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
    conn.close()
    
    if task is None:
        return jsonify({"error": "任务不存在"}), 404
    
    target_label = task['target_label']
    attack_type = task['attack_type']
    
    logging.debug(f"任务信息: 标签={target_label}, 攻击类型={attack_type}")
    
    # 尝试多种可能的路径
    possible_paths = []
    
    # 基于任务ID的路径
    possible_paths.append(f"./result/attack/{task_id}_{target_label}.png")
    
    # 传统路径
    possible_paths.append(f"./result/attack/inverted_{target_label}.png")
    
    # PIG攻击特定路径
    if attack_type == "PIG_attack" or "PIG" in attack_type:
        possible_paths.append(f"./result/PLG_MI_Inversion/success_imgs/{target_label}/0_attack_iden_{target_label}_0.png")
        possible_paths.append(f"./result/PLG_MI_Inversion/all_imgs/{target_label}/attack_iden_{target_label}_0.png")
    
    # 添加绝对路径支持
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for rel_path in possible_paths.copy():
        abs_path = os.path.join(base_dir, rel_path.lstrip('./'))
        possible_paths.append(abs_path)
    
    # 尝试所有可能的路径
    for path in possible_paths:
        logging.debug(f"尝试路径: {path}")
        if path and os.path.exists(path):
            logging.debug(f"找到图像文件: {path}")
            try:
                return send_file(path, mimetype='image/png')
            except Exception as e:
                logging.error(f"发送文件时出错: {e}")
                continue
    
    # 没有找到图像，创建一个空白图像返回
    logging.error(f"未找到任务图像，尝试了路径: {possible_paths}")
    
    # 创建一个简单的空白图像作为后备方案
    try:
        # 创建一个100x100的红色图像作为错误提示
        img = Image.new('RGB', (100, 100), color = (255, 0, 0))
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logging.error(f"创建备用图像失败: {e}")
        return jsonify({"error": "图像文件不存在且无法创建备用图像"}), 404

# 监测系统占用率
@app.route('/system-metrics', methods=['GET'])
def system_metrics():
    def generate():
        # 确保第一次调用获取基准值
        psutil.cpu_percent(interval=0.1)
        
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            has_nvml = True
        except:
            has_nvml = False
            
        while True:
            # 获取每个核心的 CPU 使用率
            per_cpu = psutil.cpu_percent(interval=1, percpu=True)
            # 获取平均 CPU 使用率
            cpu_percent = max(psutil.cpu_percent(), max(per_cpu) if per_cpu else 0)
            
            # 获取其他系统资源
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            net_io = psutil.net_io_counters()
            network_percent = min(100, (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024 / 10)
            
            # 获取 GPU 信息
            gpu_percent = 0
            gpu_memory_percent = 0
            
            if has_nvml:
                try:
                    # 假设使用第一个 GPU
                    device_count = nvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_percent = util.gpu
                        
                        # 获取 GPU 内存信息
                        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_memory_percent = (mem_info.used / mem_info.total) * 100
                except:
                    pass
            
            metrics = {
                'cpu': round(cpu_percent, 1),
                'memory': round(memory_percent, 1),
                'disk': round(disk_percent, 1),
                'network': round(network_percent, 1),
                'gpu': round(gpu_percent, 1),
                'gpu_memory': round(gpu_memory_percent, 1),
                'timestamp': time.time()
            }
            
            data = f"data: {json.dumps(metrics)}\n\n"
            yield data
            
            # 稍微减少时间间隔可以更快地反映系统变化
            time.sleep(2)
    
    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Cache-Control', 'no-cache')
    response.headers.add('Connection', 'keep-alive')
    
    return response

# 获取所有任务
@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    status_filter = request.args.get('status', 'all')
    
    conn = get_db_connection()
    if status_filter != 'all':
        tasks = conn.execute('SELECT * FROM tasks WHERE status = ? ORDER BY create_time DESC', 
                           (status_filter,)).fetchall()
    else:
        tasks = conn.execute('SELECT * FROM tasks ORDER BY create_time DESC').fetchall()
    
    conn.close()
    
    return jsonify([dict(task) for task in tasks])

# 创建新任务
@app.route("/api/tasks", methods=["POST"])
def create_task():
    data = request.json
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_db_connection()
    conn.execute('''
    INSERT INTO tasks (id, name, model, attack_type, status, progress, create_time, description, target_label, parameters)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        task_id,
        data.get('name', f"攻击任务-{task_id}"),
        data.get('model', '未指定'),
        data.get('attack_type', '模型反演'),
        'queued',
        0,
        now,
        data.get('description', ''),
        data.get('target_label', 0),
        json.dumps(data.get('parameters', {}))
    ))
    conn.commit()
    conn.close()
    
    # 将任务添加到Redis队列
    redis_client.lpush('attack_tasks', task_id)
    
    return jsonify({"id": task_id, "message": "任务创建成功"})

# 获取任务详情
@app.route("/api/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    conn = get_db_connection()
    task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
    conn.close()
    
    if task is None:
        return jsonify({"error": "任务不存在"}), 404
    
    return jsonify(dict(task))


# 预设模型的输入输出维度
MODEL_CONFIGS = {
    "MLP": {"mode": "L", "h": 112, "w": 92, "class_num": 40},
    "VGG16": {"mode": "RGB", "h": 64, "w": 64, "class_num": 1000},
    "FaceNet64": {"mode": "RGB", "h": 64, "w": 64, "class_num": 1000},
    "IR152": {"mode": "RGB", "h": 64, "w": 64, "class_num": 1000},
}

# 数据集上传API
@app.route("/api/datasets/upload", methods=["POST"])
def upload_dataset():
    """处理数据集上传请求"""
    try:
        if "dataset_file" not in request.files:
            return jsonify({"error": "未提供数据集文件"}), 400
            
        dataset_file = request.files["dataset_file"]
        if dataset_file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
            
        if not dataset_file.filename.endswith('.zip'):
            return jsonify({"error": "仅支持zip格式的数据集文件"}), 400
        
        dataset_name = request.form.get("dataset_name", "未命名数据集")
        description = request.form.get("description", "用户上传的数据集")
        
        # 处理上传的数据集
        result = dataset_manager.process_uploaded_dataset(
            dataset_file, dataset_name, description
        )
        
        return jsonify(result), 201
    
    except Exception as e:
        return jsonify({"error": f"上传数据集失败: {str(e)}"}), 500

# 获取数据集列表API
@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    """获取所有可用数据集"""
    try:
        datasets = dataset_manager.get_all_datasets()
        return jsonify(datasets)
    except Exception as e:
        return jsonify({"error": f"获取数据集失败: {str(e)}"}), 500

# 获取数据集类别图像API
@app.route("/api/datasets/<dataset_id>/images/<int:label>", methods=["GET"])
def get_dataset_image(dataset_id, label):
    """获取指定数据集和类别的代表图像"""
    try:
        image_path = dataset_manager.get_dataset_class_image(dataset_id, label)
        return send_file(image_path)
    except Exception as e:
        return jsonify({"error": f"获取图像失败: {str(e)}"}), 500

# 评估攻击效果API
@app.route("/api/evaluations/evaluate", methods=["POST"])
def evaluate_attack():
    """评估攻击效果"""
    try:
        data = request.json
        task_id = data.get("task_id")
        dataset_id = data.get("dataset_id")
        metrics = data.get("metrics", ["psnr", "ssim", "mse", "target_accuracy"])
        
        if not task_id or not dataset_id:
            return jsonify({"error": "必须提供task_id和dataset_id"}), 400
        
        result = dataset_manager.evaluate_attack_result(task_id, dataset_id, metrics)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"评估失败: {str(e)}"}), 500

@app.route("/api/datasets/by-model/<model_name>", methods=["GET"])
def get_dataset_by_model(model_name):
    """获取模型关联的数据集"""
    try:
        # 查询数据库获取与模型关联的数据集
        conn = get_db_connection()
        dataset = conn.execute('SELECT * FROM datasets WHERE model_name = ?', (model_name,)).fetchone()
        conn.close()
        
        if dataset is None:
            return jsonify({"error": f"未找到模型 {model_name} 关联的数据集"}), 404
        
        return jsonify(dict(dataset))
    except Exception as e:
        return jsonify({"error": f"获取模型数据集失败: {str(e)}"}), 500
# 获取评估结果API
@app.route("/api/evaluations/results", methods=["GET"])
def get_evaluations_results():
    """获取所有评估结果"""
    try:
        task_id = request.args.get("task_id")
        results = dataset_manager.get_evaluation_results(task_id)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"获取评估结果失败: {str(e)}"}), 500

# 获取评估图表API
@app.route("/api/evaluations/charts/<chart_type>", methods=["GET"])
def get_evaluation_charts(chart_type):
    """获取评估结果图表"""
    try:
        charts = dataset_manager.generate_evaluation_charts()
        
        if chart_type not in charts:
            return jsonify({"error": f"图表类型{chart_type}不存在"}), 404
        
        # 将base64图像转换为响应
        chart_data = charts[chart_type]
        image_data = base64.b64decode(chart_data)
        
        return Response(
            image_data,
            mimetype='image/png'
        )
    except Exception as e:
        return jsonify({"error": f"获取图表失败: {str(e)}"}), 500
@app.route("/api/datasets/upload-for-model", methods=["POST"])
def upload_dataset_for_model():
    """处理与模型关联的数据集上传请求"""
    try:
        if "dataset_file" not in request.files:
            return jsonify({"error": "未提供数据集文件"}), 400
            
        dataset_file = request.files["dataset_file"]
        if dataset_file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
            
        if not dataset_file.filename.endswith(('.zip', '.tar.gz', '.tar')):
            return jsonify({"error": "仅支持zip、tar.gz或tar格式的数据集文件"}), 400
        
        dataset_name = request.form.get("dataset_name", "未命名数据集")
        model_name = request.form.get("model_name")
        description = request.form.get("description", "与模型关联的数据集")
        
        if not model_name:
            return jsonify({"error": "必须提供关联的模型名称"}), 400
        
        # 处理上传的数据集
        result = dataset_manager.process_uploaded_dataset_for_model(
            dataset_file, dataset_name, model_name, description
        )
        
        # 使用model_scanner更新配置，而不是直接操作不存在的数据库表
        try:
            scanner = get_model_scanner()
            for config in scanner.model_configs:
                if config.model_name == model_name:
                    config.dataset_id = result["id"]
                    scanner.save_config()
                    break
        except Exception as e:
            logging.error(f"更新模型配置出错: {e}")
        
        return jsonify(result), 201
    
    except Exception as e:
        return jsonify({"error": f"上传数据集失败: {str(e)}"}), 500
# 批量评估API
@app.route("/api/evaluations", methods=["POST"])
def create_batch_evaluation():
    """创建批量评估任务"""
    try:
        data = request.json
        evaluation_id = dataset_manager.run_batch_evaluation(data)
        return jsonify({"id": evaluation_id, "message": "批量评估任务已创建"})
    except Exception as e:
        return jsonify({"error": f"创建批量评估任务失败: {str(e)}"}), 500

# 更新任务状态    
@app.route("/api/tasks/<task_id>/status", methods=["PUT"])
def update_task_status(task_id):
    data = request.json
    status = data.get('status')
    progress = data.get('progress')
    error_message = data.get('error_message')
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_db_connection()
    task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
    
    if task is None:
        conn.close()
        return jsonify({"error": "任务不存在"}), 404
    
    # 更新相应的时间字段
    if status == 'running' and task['status'] != 'running':
        conn.execute('UPDATE tasks SET start_time = ? WHERE id = ?', (now, task_id))
    
    if status in ['completed', 'failed'] and task['status'] not in ['completed', 'failed']:
        conn.execute('UPDATE tasks SET end_time = ? WHERE id = ?', (now, task_id))
    
    # 更新任务状态和进度
    if error_message:
        conn.execute('UPDATE tasks SET status = ?, progress = ?, error_message = ? WHERE id = ?', 
                   (status, progress, error_message, task_id))
    else:
        conn.execute('UPDATE tasks SET status = ?, progress = ? WHERE id = ?', 
                   (status, progress, task_id))
    
    conn.commit()
    conn.close()
    
    return jsonify({"message": "任务状态已更新"})

# 攻击模型反转攻击接口
@app.route("/attack", methods=["POST"])
def attack():
    """处理模型反演攻击请求并返回结果"""
    logging.basicConfig(level=logging.DEBUG)
    
    data = request.json

    if not data or "target_label" not in data:
        return jsonify({"error": "无效请求，需要'target_label'参数"}), 400
    
    target_label = data["target_label"]
    attack_method_name = data.get("attack_method", "standard_attack")  # 默认使用标准攻击
    target_model = data.get("target_model", "MLP")  # 默认针对MLP目标模型
    dataset = data.get("dataset", "ATT40")  # 默认使用CIFAR-10数据集
    class_num = data.get("class_num", 40)  # 默认分类数为40
    image_size = data.get("image_size", "64*64")  # 默认图像大小为224
    channels = data.get("channels", 3)  # 默认彩色图像，通道数为3
    # 解析图像大小
    h,w = image_size.split("*")
    h,w = int(h), int(w)
    print("Value of w:", w)
    # 找到相应的可用目标模型
    param_file = ""
    for key, value in matched_models.items():
        print("key is",key)
        print("value is",value)
        if key == target_model:
            param_file = value
            break
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备
    print("param_file is",param_file)
    model = load_model(target_model, param_file , device , class_num)
    
    # 加载生成模型
    G = load_G(attack_method_name, target_model, dataset, device, class_num)
    
    
    # 创建任务记录
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_db_connection()
    conn.execute('''
    INSERT INTO tasks (id, name, model, attack_type, status, progress, create_time, description, target_label)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        task_id,
        f"标签{target_label}的{attack_method_name}攻击",
        "目标分类器",
        attack_method_name,
        'running',
        0,
        now,
        f"针对标签{target_label}的反演攻击",
        target_label
    ))
    conn.commit()
    
    # 更新任务状态为运行中
    conn.execute('UPDATE tasks SET status = ?, start_time = ?, progress = ? WHERE id = ?', 
               ('running', now, 10, task_id))
    conn.commit()
    conn.close()
    
    try:
        logging.debug(f"开始执行攻击 方法:{attack_method_name} 目标:{target_label} 任务ID:{task_id}")
        
        # 确保结果目录存在
        result_dir = "./result/attack"
        os.makedirs(result_dir, exist_ok=True)
        
        # 执行攻击，传入任务ID
        result_image = reconstruct(attack_method_name, model, G, target_label,  h, w, channels, device, task_id)
        logging.debug(f"攻击完成，结果类型: {type(result_image)}")
        
        # 更新任务进度
        conn = get_db_connection()
        conn.execute('UPDATE tasks SET progress = ? WHERE id = ?', (50, task_id))
        conn.commit()
        conn.close()
        
        # 处理结果图像
        image_path = ""
        image_base64 = ""
        
        if result_image:
            # 如果返回的是文件路径
            if isinstance(result_image, str) and os.path.isfile(result_image):
                logging.debug(f"结果是文件路径: {result_image}")
                # 使用任务ID生成唯一的文件名
                final_path = f"./result/attack/{task_id}_{target_label}.png"
                image_path = f"/static/result/attack/{task_id}_{target_label}.png"
                
                # 如果路径不是预期的位置，复制到标准位置
                if result_image != final_path:
                    import shutil
                    shutil.copyfile(result_image, final_path)
                
                # 读取文件并转为base64
                with open(final_path, "rb") as img_file:
                    image_data = img_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 如果已经是base64字符串
            elif isinstance(result_image, str) and (
                result_image.startswith("iVBOR") or 
                result_image.startswith("/9j/") or
                result_image.startswith("data:")
            ):
                logging.debug(f"结果是base64字符串，长度: {len(result_image)}")
                # 移除data:image前缀（如果有）
                if result_image.startswith("data:"):
                    image_base64 = result_image.split(",", 1)[1]
                else:
                    image_base64 = result_image
                
                # 使用任务ID保存到文件以便后续访问
                final_path = f"./result/attack/{task_id}_{target_label}.png"
                image_path = f"/static/result/attack/{task_id}_{target_label}.png"
                
                try:
                    image_data = base64.b64decode(image_base64)
                    with open(final_path, "wb") as img_file:
                        img_file.write(image_data)
                except Exception as e:
                    logging.error(f"保存base64图像失败: {e}")
            
            # 如果是其他格式（如tensor或numpy数组）
            elif hasattr(result_image, 'numpy') or str(type(result_image)).find('numpy') != -1 or str(type(result_image)).find('torch') != -1:
                logging.debug(f"结果是tensor或数组类型")
                # 保存到文件
                final_path = f"./result/attack/{task_id}_{target_label}.png"
                image_path = f"/static/result/attack/{task_id}_{target_label}.png"
                
                try:
                    # 尝试使用PIL保存
                    from PIL import Image
                    import numpy as np
                    
                    # 如果是PyTorch tensor
                    if hasattr(result_image, 'cpu') and hasattr(result_image, 'detach'):
                        result_image = result_image.detach().cpu().numpy()
                    
                    # 如果是numpy数组，确保格式正确
                    if hasattr(result_image, 'shape'):
                        # 归一化到0-255
                        if result_image.max() <= 1.0:
                            result_image = (result_image * 255).astype(np.uint8)
                        else:
                            result_image = result_image.astype(np.uint8)
                        
                        # 确保数组形状正确 (H,W,C) 或 (H,W)
                        if len(result_image.shape) == 3 and result_image.shape[0] <= 3:
                            # 如果是 (C,H,W) 格式，转换为 (H,W,C)
                            result_image = np.transpose(result_image, (1, 2, 0))
                        
                        img = Image.fromarray(result_image)
                        img.save(final_path)
                        
                        # 转换为base64
                        with open(final_path, "rb") as img_file:
                            image_data = img_file.read()
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                    else:
                        logging.error("无法识别的图像数组格式")
                except Exception as e:
                    logging.error(f"保存tensor/numpy图像失败: {e}")
            else:
                logging.error(f"未知的结果类型: {type(result_image)}")
        
        # 更新任务状态及图像路径
        conn = get_db_connection()
        if image_path or image_base64:
            conn.execute('UPDATE tasks SET status = ?, end_time = ?, progress = ?, image_path = ? WHERE id = ?', 
                       ('completed', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100, image_path, task_id))
            
            response_data = {
                "message": "攻击成功", 
                "task_id": task_id
            }
            
            # 添加图像数据到响应
            if image_base64:
                response_data["image"] = image_base64
            if image_path:
                response_data["image_path"] = image_path
                
            conn.commit()
            conn.close()
            return jsonify(response_data), 200
        else:
            # 攻击失败 - 没有生成有效图像
            conn.execute('UPDATE tasks SET status = ?, end_time = ?, error_message = ? WHERE id = ?', 
                       ('failed', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "攻击未生成有效图像", task_id))
            conn.commit()
            conn.close()
            return jsonify({"error": "攻击未生成有效图像", "task_id": task_id}), 500
            
    except Exception as e:
        # 捕获所有异常
        error_message = str(e)
        logging.error(f"攻击过程中发生错误: {error_message}")
        
        # 更新任务状态为失败
        conn = get_db_connection()
        conn.execute('UPDATE tasks SET status = ?, end_time = ?, error_message = ? WHERE id = ?', 
                   ('failed', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message, task_id))
        conn.commit()
        conn.close()
        
        return jsonify({"error": "内部服务器错误", "message": error_message, "task_id": task_id}), 500


def get_upload_path(filename):
    """根据文件类型返回存储路径"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in {'.tar', '.pkl', '.pth'}:
        return os.path.join(CHECKPOINT_DIR, filename)
    elif ext == '.py':
        return os.path.join(MODEL_DIR, filename)
    else:
        return None  # 不支持的文件类型


#################################################
# 新增的评估API端点
#################################################

# 引入必要的库
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:
    # 如果没有安装skimage库，提供一个简单的替代函数
    def peak_signal_noise_ratio(img1, img2):
        return 0.0
    
    def structural_similarity(img1, img2):
        return 0.0
    logging.warning("scikit-image库未安装，图像质量评估将返回默认值0")

def tensor_to_native(obj):
    """将PyTorch Tensor转换为可JSON序列化的Python原生类型"""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_native(i) for i in obj]
    else:
        return obj

# 创建批量评估任务API
@app.route("/api/evaluations", methods=["POST"])
def create_evaluation():
    data = request.json
    evaluation_id = f"eval-{uuid.uuid4().hex[:8]}"
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 计算总样本数
    attack_methods = data.get('attackMethods', [])
    label_start = data.get('labelRange', {}).get('start', 0)
    label_end = data.get('labelRange', {}).get('end', 9)
    samples_per_label = data.get('samplesPerLabel', 10)
    total_samples = len(attack_methods) * (label_end - label_start + 1) * samples_per_label
    
    # 存储评估任务信息
    conn = get_db_connection()
    conn.execute('''
    INSERT INTO evaluations (id, name, dataset, target_model, attack_methods, status, progress, 
                            total_samples, completed_samples, start_time, parameters, create_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        evaluation_id,
        f"评估任务-{evaluation_id}",
        data.get('dataset', 'att_faces'),
        data.get('targetModel', 'MLP'),
        json.dumps(attack_methods),
        'running',
        0,
        total_samples,
        0,
        now,
        json.dumps(data),
        now
    ))
    conn.commit()
    conn.close()
    
    # 启动后台线程执行评估
    thread = Thread(target=run_evaluation, args=(evaluation_id, data))
    thread.daemon = True
    thread.start()
    
    return jsonify({"id": evaluation_id, "message": "评估任务已创建"})

# 获取评估任务列表
@app.route("/api/evaluations", methods=["GET"])
def get_evaluations():
    conn = get_db_connection()
    evaluations = conn.execute('SELECT * FROM evaluations ORDER BY create_time DESC').fetchall()
    conn.close()
    
    return jsonify([dict(eval_task) for eval_task in evaluations])

# 获取评估任务详情
@app.route("/api/evaluations/<evaluation_id>", methods=["GET"])
def get_evaluation(evaluation_id):
    conn = get_db_connection()
    evaluation = conn.execute('SELECT * FROM evaluations WHERE id = ?', (evaluation_id,)).fetchone()
    conn.close()
    
    if evaluation is None:
        return jsonify({"error": "评估任务不存在"}), 404
    
    result = dict(evaluation)
    # 解析JSON字段
    for field in ['attack_methods', 'parameters', 'results']:
        if result.get(field):
            try:
                result[field] = json.loads(result[field])
            except:
                pass
    
    return jsonify(result)

# 停止评估任务
@app.route("/api/evaluations/<evaluation_id>/stop", methods=["POST"])
def stop_evaluation(evaluation_id):
    conn = get_db_connection()
    conn.execute('UPDATE evaluations SET status = ? WHERE id = ?', ('stopped', evaluation_id))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "评估任务已停止"})


@app.route('/api/models/available', methods=['GET'])
def get_available_model_types():
    """获取所有可用的模型类型"""
    scanner = get_model_scanner()
    model_defs, _, _ = scanner.scan()
    return jsonify(model_defs)

@app.route('/api/models/parameters', methods=['GET'])
def get_available_parameters():
    """获取所有可用的模型参数文件"""
    scanner = get_model_scanner()
    _, param_files, _ = scanner.scan()
    return jsonify(param_files)

@app.route('/api/models/configurations', methods=['GET'])
def get_model_configurations():
    """获取所有的模型配置"""
    scanner = get_model_scanner()
    _, _, configs = scanner.scan()
    return jsonify([config.to_dict() for config in configs])

# 修改模型配置创建API
@app.route('/api/models/configurations', methods=['POST'])
def create_model_configuration():
    """创建新的模型配置"""
    data = request.json
    
    # 验证必要的字段
    required_fields = ['model_name', 'param_file']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'缺少必要字段: {field}'}), 400
    
    # 创建配置对象，ModelConfig类会自动生成model_id
    config = ModelConfig(
        model_name=data['model_name'],
        param_file=data['param_file'],
        class_num=data.get('class_num', 0),
        input_shape=tuple(data.get('input_shape', (0, 0))),
        model_type=data.get('model_type', data['model_name'])
    )
    
    # 添加配置
    scanner = get_model_scanner()
    result = scanner.add_config(config)
    
    if result:
        return jsonify({'message': '模型配置创建成功', 'config': config.to_dict()}), 201
    else:
        return jsonify({'error': '模型配置创建失败，请检查模型和参数文件是否兼容'}), 400

@app.route('/api/models/configurations/<model_name>/<param_file>', methods=['DELETE'])
def delete_model_configuration(model_name, param_file):
    """删除模型配置"""
    scanner = get_model_scanner()
    result = scanner.remove_config(model_name, param_file)
    
    if result:
        return jsonify({'message': '模型配置删除成功'}), 200
    else:
        return jsonify({'error': '未找到指定的模型配置'}), 404

@app.route('/api/models/scan', methods=['POST'])
def trigger_model_scan():
    """触发模型扫描，这对于需要手动刷新的情况很有用"""
    scanner = get_model_scanner()
    model_defs, param_files, configs = scanner.scan()
    
    return jsonify({
        'model_definitions': model_defs,
        'parameter_files': param_files,
        'configurations': [config.to_dict() for config in configs],
        'message': f'扫描完成，发现 {len(model_defs)} 个模型定义，{len(param_files)} 个参数文件，{len(configs)} 个配置'
    }), 200

@app.route('/api/models/validate-config', methods=['POST'])
def validate_model_config():
    """验证模型配置是否有效"""
    data = request.json
    
    # 验证必要的字段
    required_fields = ['model_name', 'param_file']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'缺少必要字段: {field}'}), 400
    
    # 创建配置对象
    config = ModelConfig(
        model_name=data['model_name'],
        param_file=data['param_file'],
        class_num=data.get('class_num', 0),
        input_shape=tuple(data.get('input_shape', (0, 0))),
        model_type=data.get('model_type', data['model_name'])
    )
    
    # 验证配置
    scanner = get_model_scanner()
    valid = scanner.validate_config(config)
    
    if not valid:
        return jsonify({'valid': False, 'message': '模型或参数文件不存在'}), 400
    
    # 尝试加载模型
    load_success = scanner.try_load_model(config)
    
    if load_success:
        return jsonify({'valid': True, 'message': '配置有效，模型加载成功'}), 200
    else:
        return jsonify({'valid': False, 'message': '模型加载失败，配置可能无效'}), 400

@app.route('/api/models/auto-detect-config', methods=['POST'])
def auto_detect_model_config():
    """自动检测参数文件的配置信息"""
    data = request.json
    
    if 'param_file' not in data:
        return jsonify({'error': '缺少必要字段: param_file'}), 400
    
    param_file = data['param_file']
    
    # 获取扫描器
    scanner = get_model_scanner()
    
    # 确保刷新参数文件列表
    _, param_files, _ = scanner.scan()
    
    if param_file not in param_files:
        return jsonify({'error': f'参数文件不存在: {param_file}'}), 404
    
    # 尝试猜测模型
    model_name = scanner.guess_model_type(param_file)
    
    if not model_name:
        return jsonify({'error': '无法自动识别适合此参数文件的模型类型'}), 400
    
    # 构建建议的配置
    suggested_config = None
    
    if model_name == "MLP":
        suggested_config = {
            "model_name": model_name,
            "param_file": param_file,
            "class_num": 40,  # AT&T Faces默认40类
            "input_shape": [112, 92],  # MLP默认输入形状
            "model_type": "MLP"
        }
    elif model_name in ["VGG16", "FaceNet64", "IR152"]:
        suggested_config = {
            "model_name": model_name,
            "param_file": param_file,
            "class_num": 1000,  # CelebA默认1000类
            "input_shape": [64, 64],  # 默认输入形状
            "model_type": model_name
        }
    else:
        suggested_config = {
            "model_name": model_name,
            "param_file": param_file,
            "class_num": 10,  # 默认10类
            "input_shape": [64, 64],  # 默认输入形状
            "model_type": model_name
        }
    
    return jsonify({
        'message': f'自动检测到适合的模型类型: {model_name}',
        'suggested_config': suggested_config
    }), 200
@app.route('/api/models/search', methods=['POST'])
def search_model():
    """根据模型名称搜索模型结构和参数文件"""
    data = request.json
    model_name = data.get('model_name', '').strip()
    
    if not model_name:
        return jsonify({'error': '请提供模型名称'}), 400
    
    scanner = get_model_scanner()
    model_defs, param_files, configs = scanner.scan()
    
    # 检查是否存在模型结构
    structure_found = False
    for name in model_defs.keys():
        if name.lower() == model_name.lower() or name.lower().startswith(model_name.lower()):
            structure_found = True
            model_name = name  # 使用找到的精确名称
            break
    
    # 检查是否存在匹配的参数文件
    matching_parameters = []
    for param_file in param_files:
        param_lower = param_file.lower()
        if model_name.lower() in param_lower:
            matching_parameters.append(param_file)
    
    # 检查是否已经有配置
    existing_config = None
    config_id = None
    for config in configs:
        if config.model_name.lower() == model_name.lower():
            existing_config = config
            config_id = config.model_id if hasattr(config, 'model_id') else f"{config.model_name}-{config.param_file}"
            break
    
    # 如果找到结构和参数但没有配置，尝试创建自动配置建议
    auto_config = None
    if structure_found and matching_parameters and not existing_config:
        # 选择第一个匹配的参数文件
        param_file = matching_parameters[0]
        
        if model_name == "MLP":
            auto_config = {
                "model_name": model_name,
                "param_file": param_file,
                "class_num": 40,  # AT&T Faces默认40类
                "input_shape": [112, 92],  # MLP默认输入形状
                "model_type": model_name
            }
        elif model_name in ["VGG16", "FaceNet64", "IR152"]:
            auto_config = {
                "model_name": model_name,
                "param_file": param_file,
                "class_num": 1000,  # CelebA默认1000类
                "input_shape": [64, 64],  # 默认输入形状
                "model_type": model_name
            }
        else:
            auto_config = {
                "model_name": model_name,
                "param_file": param_file,
                "class_num": 10,  # 默认10类
                "input_shape": [64, 64],  # 默认输入形状
                "model_type": model_name
            }
    
    return jsonify({
        'model_name': model_name,
        'structure_found': structure_found,
        'parameters_found': len(matching_parameters) > 0,
        'matching_parameters': matching_parameters,
        'id': config_id,
        'auto_config': auto_config
    })

# 添加获取已配置模型的API
@app.route('/api/models/configured', methods=['GET'])
def get_configured_models():
    """获取所有已配置的模型，供攻击页面使用"""
    scanner = get_model_scanner()
    _, _, configs = scanner.scan()
    
    configured_models = []
    for config in configs:
        model_id = config.model_id if hasattr(config, 'model_id') else f"{config.model_name}-{config.param_file}"
        model_data = {
            "id": model_id,
            "name": f"{config.model_name}-{config.class_num}类",
            "model_name": config.model_name,
            "param_file": config.param_file,
            "class_num": config.class_num,
            "input_shape": list(config.input_shape) if isinstance(config.input_shape, tuple) else config.input_shape
        }
        configured_models.append(model_data)
    
    return jsonify(configured_models)
# 修改模型预测端点，使用配置的模型
@app.route("/predict", methods=["POST"])
def predict():
    """使用已配置的模型进行预测"""
    # 接收图像文件
    if "image_file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # 获取参数
    model_name = request.form.get("model_name", type=str)
    param_file = request.form.get("param_file", type=str)
    class_num = request.form.get("class_num", type=int, default=40)
    
    # 如果没有提供模型名称和参数文件，使用默认模型
    if not model_name or not param_file:
        scanner = get_model_scanner()
        _, _, configs = scanner.scan()
        
        if not configs:
            return jsonify({"error": "没有可用的模型配置，请先配置模型"}), 400
        
        # 使用第一个可用的配置
        config = configs[0]
        model_name = config.model_name
        param_file = config.param_file
        class_num = config.class_num
    
    try:
        # 确定设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        scanner = get_model_scanner()
        config = ModelConfig(
            model_name=model_name,
            param_file=param_file,
            class_num=class_num
        )
        
        if not scanner.validate_config(config):
            return jsonify({"error": "模型配置无效"}), 400
        
        # 以下部分保持不变，使用现有的load_model和image_to_tensor函数
        from upload_importlib import load_model, image_to_tensor
        model = load_model(model_name, param_file, device, class_num)
        
        # 处理图像
        image_file = request.files["image_file"]
        image_tensor = image_to_tensor(image_file).to(device)
        
        # 执行预测
        with torch.no_grad():
            output = model.predict(image_tensor)
            confidences = torch.softmax(output, dim=-1).squeeze(0)
            prediction = torch.argmax(confidences).item()
        
        # 返回结果
        return jsonify({
            "prediction": prediction, 
            "confidences": confidences.tolist(),
            "model_info": {
                "model_name": model_name,
                "param_file": param_file,
                "class_num": class_num
            }
        })
    except Exception as e:
        return jsonify({"error": f"预测出错: {str(e)}"}), 500

# 更新文件上传端点
@app.route('/checkpoint', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        ext = filename.rsplit('.', 1)[1].lower()

        # 根据文件扩展名选择上传目录
        if ext == 'py':
            upload_dir = app.config['PYTHON_FOLDER']
        else:
            upload_dir = app.config['CTP_FOLDER']

        # 如果文件夹不存在，创建
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        upload_path = os.path.join(upload_dir, filename)
        file.save(upload_path)

        # 扫描模型文件
        scanner = get_model_scanner()
        scanner.scan()

        # 尝试自动识别模型类型
        model_name = scanner.guess_model_type(filename)
        auto_config_info = {}

        if model_name:
            auto_config_info["detected_model"] = model_name
            auto_config_info["message"] = f"检测到可能匹配的模型: {model_name}"

        return jsonify({
            "message": f"File {filename} uploaded successfully",
            "auto_detection": auto_config_info
        }), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

# 在应用启动时初始化模型扫描器
# 替换 @app.before_first_request 装饰器
def initialize_model_scanner():
    # 获取模型扫描器并执行初始扫描
    scanner = get_model_scanner()
    scanner.scan()

# 在应用启动时执行初始化
with app.app_context():
    initialize_model_scanner()




# 下载评估报告API
@app.route("/api/evaluations/<evaluation_id>/report", methods=["GET"])
def download_evaluation_report(evaluation_id):
    try:
        conn = get_db_connection()
        evaluation = conn.execute('SELECT * FROM evaluations WHERE id = ?', (evaluation_id,)).fetchone()
        conn.close()
        
        if evaluation is None:
            return jsonify({"error": "评估任务不存在"}), 404
        
        # 生成简单的HTML报告
        eval_dict = dict(evaluation)
        results = json.loads(eval_dict.get('results', '[]')) if eval_dict.get('results') else []
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>评估报告 - {eval_dict['id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .header {{ margin-bottom: 20px; }}
                .section {{ margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>模型攻击评估报告</h1>
                <p>评估ID: {eval_dict['id']}</p>
                <p>创建时间: {eval_dict['create_time']}</p>
                <p>状态: {eval_dict['status']}</p>
            </div>
            
            <div class="section">
                <h2>评估配置</h2>
                <table>
                    <tr><th>数据集</th><td>{eval_dict['dataset']}</td></tr>
                    <tr><th>目标模型</th><td>{eval_dict['target_model']}</td></tr>
                    <tr><th>攻击方法</th><td>{eval_dict['attack_methods']}</td></tr>
                    <tr><th>样本总数</th><td>{eval_dict['total_samples']}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>评估结果</h2>
                <table>
                    <tr>
                        <th>攻击方法</th>
                        <th>准确率</th>
                        <th>攻击成功率</th>
                        <th>PSNR</th>
                        <th>SSIM</th>
                        <th>平均置信度</th>
                        <th>执行时间(秒)</th>
                    </tr>
        """
        
        for result in results:
            html_content += f"""
                    <tr>
                        <td>{result.get('method', '-')}</td>
                        <td>{result.get('accuracy', 0) * 100:.2f}%</td>
                        <td>{result.get('successRate', 0) * 100:.2f}%</td>
                        <td>{result.get('psnr', 0):.2f}</td>
                        <td>{result.get('ssim', 0):.3f}</td>
                        <td>{result.get('avgConfidence', 0) * 100:.2f}%</td>
                        <td>{result.get('executionTime', 0):.0f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # 创建内存文件并写入HTML内容
        report_io = io.BytesIO()
        report_io.write(html_content.encode('utf-8'))
        report_io.seek(0)
        
        return send_file(
            report_io, 
            mimetype='text/html',
            as_attachment=True,
            download_name=f"evaluation_report_{evaluation_id}.html"
        )
    
    except Exception as e:
        logging.error(f"生成评估报告时出错: {e}")
        return jsonify({"error": f"生成评估报告失败: {str(e)}"}), 500

# 后台执行评估的函数
def run_evaluation(evaluation_id, config):
    try:
        # 解析配置
        dataset = config.get('dataset', 'att_faces')
        target_model = config.get('targetModel', 'MLP')
        attack_methods = config.get('attackMethods', [])
        label_start = config.get('labelRange', {}).get('start', 0)
        label_end = config.get('labelRange', {}).get('end', 9)
        samples_per_label = config.get('samplesPerLabel', 10)
        advanced_settings = config.get('advancedSettings', {})
        
        total_samples = len(attack_methods) * (label_end - label_start + 1) * samples_per_label
        completed_samples = 0
        results = []
        
        start_time = time.time()
        
        # 对每种攻击方法进行评估
        for method in attack_methods:
            method_results = {
                'method': method,
                'model': target_model,
                'dataset': dataset,
                'accuracy': 0,
                'successRate': 0,
                'psnr': 0,
                'ssim': 0,
                'avgConfidence': 0,
                'executionTime': 0,
                'classSuccessRates': [],
                'sampleCount': 0,
                'completedSamples': 0,
                'confidenceDistribution': []
            }
            
            method_start_time = time.time()
            success_count = 0
            total_psnr = 0
            total_ssim = 0
            total_confidence = 0
            class_success_counts = [0] * (label_end - label_start + 1)
            confidence_distribution = [0] * 10  # 10个置信度区间
            
            # 更新当前正在处理的方法
            conn = get_db_connection()
            conn.execute('''
            UPDATE evaluations SET 
                parameters = ?,
                progress = ?
            WHERE id = ?
            ''', (
                json.dumps({
                    **json.loads(conn.execute('SELECT parameters FROM evaluations WHERE id = ?', 
                                             (evaluation_id,)).fetchone()[0]),
                    'currentMethod': method,
                    'currentLabel': label_start
                }),
                int(completed_samples * 100 / total_samples),
                evaluation_id
            ))
            conn.commit()
            conn.close()
            
            # 对每个标签进行评估
            for label in range(label_start, label_end + 1):
                # 检查任务是否被停止
                conn = get_db_connection()
                status = conn.execute('SELECT status FROM evaluations WHERE id = ?', 
                                     (evaluation_id,)).fetchone()[0]
                conn.close()
                
                if status == 'stopped':
                    return
                
                # 更新当前标签
                conn = get_db_connection()
                conn.execute('''
                UPDATE evaluations SET 
                    parameters = ?
                WHERE id = ?
                ''', (
                    json.dumps({
                        **json.loads(conn.execute('SELECT parameters FROM evaluations WHERE id = ?', 
                                                 (evaluation_id,)).fetchone()[0]),
                        'currentLabel': label
                    }),
                    evaluation_id
                ))
                conn.commit()
                conn.close()
                
                # 对每个样本进行攻击和评估
                for i in range(samples_per_label):
                    # 执行攻击
                    try:
                        # 生成一个临时任务ID用于此次攻击
                        temp_task_id = f"eval-{evaluation_id}-{method}-{label}-{i}"
                        
                        # 执行攻击
                        if method == "standard_attack":
                            from attack.standard_attack import standard_attack
                            image_data = standard_attack(label, temp_task_id)
                        elif method == "PIG_attack":
                            from backend.attack.PIG_attack1 import PIG_attack
                            # 添加参数以减少批次和种子数量
                            image_data = PIG_attack(label, temp_task_id, batch_num=1, num_seeds=1, iter_times=200)
                        else:
                            # 默认使用标准攻击
                            from attack.standard_attack import standard_attack
                            image_data = standard_attack(label, temp_task_id)
                        
                        # 将base64图像转为PIL图像
                        if isinstance(image_data, str):
                            # 如果是base64字符串
                            if image_data.startswith("data:"):
                                image_data = image_data.split(",", 1)[1]
                            
                            image_bytes = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(image_bytes)).convert("L")
                        else:
                            # 如果直接返回的是PIL图像对象
                            image = image_data
                        

                        #添加根据数据集来确定h和w和class_num,即模型的输入输出维度，不过实际上应该还要由模型架构决定，这里模型后三者写死了
                        if dataset =="att_faces":
                            h, w, class_num=112, 92, 40
                        elif dataset == "celeba":
                            h, w, class_num=64, 64, 1000
                        """还有ffhq和facescrub两种数据集，这里后面有需要再补"""


                        # 执行预测
                        prediction, confidences = predict_target_model(image,target_model, h, w, class_num)
                        if hasattr(prediction, 'item'):
                            prediction = prediction.item()

                        # 计算成功率和置信度
                        is_success = (prediction == label)
                        if isinstance(confidences, torch.Tensor):
                            if hasattr(prediction, 'item'):
                                pred_idx = prediction.item()
                            else:
                                pred_idx = prediction
                            confidence = confidences[pred_idx].item()
                        else:
                            confidence = confidences[prediction]
                        
                        if is_success:
                            success_count += 1
                            class_success_counts[label - label_start] += 1
                        
                        total_confidence += confidence
                        
                        # 确定置信度区间
                        confidence_bin = min(9, int(confidence * 10))
                        confidence_distribution[confidence_bin] += 1

                        # 尝试计算PSNR和SSIM (需要原始图像)
                        try:
                            # 添加调试信息
                            logging.debug(f"图像类型: {type(image)}, 形状: {np.array(image).shape if hasattr(image, 'shape') else '未知'}")
                            logging.debug(f"图像数据范围: 最小值={np.min(np.array(image))}, 最大值={np.max(np.array(image))}")
                            
                            # 获取参考图像 - 使用真实参考图像
                            # 方法1：使用填充值为中间值的图像
                            reference_image = np.ones((112, 92)) * 128
                            
                            # 将攻击图像转换为numpy数组
                            attack_image = np.array(image)
                            
                            # 确保图像大小匹配
                            if attack_image.shape != (112, 92):
                                # 如果形状不匹配，尝试调整大小
                                temp_img = Image.fromarray(attack_image)
                                temp_img = temp_img.resize((92, 112))
                                attack_image = np.array(temp_img)
                            
                            # 确保两个图像都是uint8类型且在0-255范围内
                            reference_image = reference_image.astype(np.uint8)
                            
                            # 重新检查攻击图像的数据范围
                            if attack_image.max() <= 1.0 and attack_image.min() >= 0:
                                attack_image = (attack_image * 255).astype(np.uint8)
                            elif attack_image.max() > 255:
                                attack_image = ((attack_image / attack_image.max()) * 255).astype(np.uint8)
                            else:
                                attack_image = attack_image.astype(np.uint8)
                            
                            # 确认两个图像的格式相同
                            logging.debug(f"参考图像: 形状={reference_image.shape}, 类型={reference_image.dtype}, 范围=[{reference_image.min()}-{reference_image.max()}]")
                            logging.debug(f"攻击图像: 形状={attack_image.shape}, 类型={attack_image.dtype}, 范围=[{attack_image.min()}-{attack_image.max()}]")
                            
                            # 计算图像质量指标
                            if attack_image.shape == reference_image.shape:
                                psnr = peak_signal_noise_ratio(reference_image, attack_image, data_range=255)
                                ssim = structural_similarity(reference_image, attack_image, data_range=255)
                                
                                logging.debug(f"成功计算图像质量指标: PSNR={psnr}, SSIM={ssim}")
                                
                                total_psnr += psnr
                                total_ssim += ssim
                            else:
                                logging.error(f"图像形状不匹配: 参考图像={reference_image.shape}, 攻击图像={attack_image.shape}")
                        except Exception as e:
                            logging.error(f"计算图像质量指标时出错: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                        
                        # 更新完成样本数
                        completed_samples += 1
                        
                        # 更新评估任务状态
                        conn = get_db_connection()
                        conn.execute('''
                        UPDATE evaluations SET 
                            completed_samples = ?, 
                            progress = ?
                        WHERE id = ?
                        ''', (
                            completed_samples,
                            int(completed_samples * 100 / total_samples),
                            evaluation_id
                        ))
                        conn.commit()
                        conn.close()
                        
                    except Exception as e:
                        logging.error(f"评估过程中出错: {e}")
                        # 仍然增加完成样本数，以便进度继续更新
                        completed_samples += 1
            
            # 计算该方法的统计结果
            sample_count = (label_end - label_start + 1) * samples_per_label
            if sample_count > 0:
                method_results['accuracy'] = success_count / sample_count
                method_results['successRate'] = success_count / sample_count
                method_results['psnr'] = total_psnr / sample_count if total_psnr > 0 else 0
                method_results['ssim'] = total_ssim / sample_count if total_ssim > 0 else 0
                method_results['avgConfidence'] = total_confidence / sample_count
            
            method_results['executionTime'] = time.time() - method_start_time
            
            # 计算每个类别的成功率
            method_results['classSuccessRates'] = [
                count / samples_per_label if samples_per_label > 0 else 0
                for count in class_success_counts
            ]
            
            # 构建置信度分布
            method_results['confidenceDistribution'] = [
                {
                    'confidenceRange': f"{i/10:.1f}-{(i+1)/10:.1f}",
                    'count': count
                }
                for i, count in enumerate(confidence_distribution)
            ]
            
            method_results['sampleCount'] = sample_count
            method_results['completedSamples'] = sample_count
            
            # 添加方法结果
            results.append(method_results)
            results = tensor_to_native(results)
            
            # 更新评估任务结果
            conn = get_db_connection()
            conn.execute('''
            UPDATE evaluations SET 
                results = ?
            WHERE id = ?
            ''', (json.dumps(results), evaluation_id))
            conn.commit()
            conn.close()
        
        # 评估完成，更新状态
        conn = get_db_connection()
        conn.execute('''
        UPDATE evaluations SET 
            status = ?, 
            end_time = ?, 
            progress = 100
        WHERE id = ?
        ''', ('completed', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), evaluation_id))
        conn.commit()
        conn.close()
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logging.error(f"评估任务执行出错: {e}")
        logging.error(f"错误详细堆栈: \n{error_trace}")
        # 更新任务状态为失败
        try:
            conn = get_db_connection()
            current_params = conn.execute('SELECT parameters FROM evaluations WHERE id = ?', 
                                        (evaluation_id,)).fetchone()
            
            if current_params and current_params[0]:
                params = json.loads(current_params[0])
                params['error'] = str(e)
                
                conn.execute('''
                UPDATE evaluations SET 
                    status = ?, 
                    end_time = ?, 
                    parameters = ?
                WHERE id = ?
                ''', (
                    'failed', 
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    json.dumps(params),
                    evaluation_id
                ))
            else:
                conn.execute('''
                UPDATE evaluations SET 
                    status = ?, 
                    end_time = ?
                WHERE id = ?
                ''', (
                    'failed', 
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    evaluation_id
                ))
                
            conn.commit()
            conn.close()
        except Exception as db_error:
            logging.error(f"更新评估失败状态时出错: {db_error}")

# 全局变量存储匹配的模型和参数即可用的模型
matched_models = {}
# 返回所有可用的模型函数
@app.route('/models', methods=['GET'])
def get_models_and_checkpoints():
    """ 获取所有可用的模型和已上传的模型参数，并匹配对应关系 """
    models = get_available_models()
    checkpoints = get_available_params()

    global matched_models
    matched_models = {}

    for checkpoint in checkpoints:
        checkpoint_prefix = checkpoint.split("_")[0]  # 取 _ 之前的部分
        for key, value in models.items():
            if value == checkpoint_prefix:
                matched_models[key] = checkpoint  # 存入匹配的模型

    return jsonify({
        "models": models,
        "checkpoints": checkpoints,
        "matched_models": matched_models  # 返回匹配结果
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True,use_reloader=False)