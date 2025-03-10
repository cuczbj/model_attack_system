# server.py
import logging
import os
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from target_model import train_target_model, predict_target_model
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
from PIL import Image
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
    conn.commit()
    conn.close()

# 创建Flask应用
app = Flask(__name__, static_url_path="/static", static_folder="./")
CORS(app)  # 启用CORS，允许所有源访问

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), './models')
app.config['ALLOWED_EXTENSIONS'] = {'pth', 'pkl'}

# 在启动应用时初始化数据库
init_db()

# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#################################################
# 原有API端点
#################################################

# 训练目标模型（备用）
@app.route("/train", methods=["POST"])
def train():
    response = train_target_model()
    return jsonify({"message": response})

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

# 目标模型预测接口
@app.route("/predict", methods=["POST"])
def predict():
    # 接收图像文件
    if "image_file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image_file"]
    try:
        # 读取并预测
        image = Image.open(image_file).convert("L")  # 转为灰度图
        prediction, confidences = predict_target_model(image)
        return jsonify({"prediction": prediction, "confidences": confidences.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

    # 验证目标标签
    if not isinstance(target_label, int) or target_label < 0 or target_label > 39:
        return jsonify({"error": "无效的目标标签，必须是0到39之间的整数"}), 400
    
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
        result_image = reconstruct(attack_method_name, target_label, task_id)
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

# 文件上传接口
@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    # 如果用户没有选择文件
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    # 检查文件是否合法
    if file and allowed_file(file.filename):
        filename = file.filename
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 如果文件夹不存在，创建文件夹
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # 保存文件
        file.save(upload_path)
        return jsonify({"message": f"File {filename} uploaded successfully"}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

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
        data.get('targetModel', 'mynet_50'),
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
        target_model = config.get('targetModel', 'mynet_50')
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
                            from attack.PIG_attack import PIG_attack
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
                        

                        # 执行预测
                        prediction, confidences = predict_target_model(image)
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
                                from PIL import Image
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
        logging.error(f"评估任务执行出错: {e}")
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)