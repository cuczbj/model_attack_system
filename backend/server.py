
import logging
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # 导入 CORS
from target_model import train_target_model,predict_target_model
from reconstruct import reconstruct
from PIL import Image
import psutil
import json
import time
from flask import Response, stream_with_context
import sqlite3
import redis
from datetime import datetime
import uuid


# 配置数据库
def get_db_connection():
    conn = sqlite3.connect('attack_system.db')
    conn.row_factory = sqlite3.Row
    return conn

# 配置Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 在server.py中修改init_db函数
def init_db():
    conn = get_db_connection()
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
    conn.commit()
    conn.close()

# 在启动应用时初始化数据库
init_db()
# 静态文件配置

app = Flask(__name__, static_url_path="/static", static_folder="./")
CORS(app)  # 启用 CORS，允许所有源访问

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), './models')  # 上传路径设置为根目录下的 models 文件夹
app.config['ALLOWED_EXTENSIONS'] = {'pth','pkl'}
# 检查文件类型
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 训练目标模型,无调用，备用
@app.route("/train", methods=["POST"])
def train():
    response = train_target_model()
    return jsonify({"message": response})

@app.route("/api/tasks/<task_id>/image", methods=["GET"])
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
        from PIL import Image
        import io
        
        # 创建一个100x100的红色图像作为错误提示
        img = Image.new('RGB', (100, 100), color = (255, 0, 0))
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logging.error(f"创建备用图像失败: {e}")
        return jsonify({"error": "图像文件不存在且无法创建备用图像"}), 404
#监测占用率
@app.route('/system-metrics', methods=['GET'])
def system_metrics():
    def generate():
        import psutil
        import json
        import time
        
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

# 获取任务状态    
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
    import base64
    import os
    import logging
    from datetime import datetime
    import uuid
    
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
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)