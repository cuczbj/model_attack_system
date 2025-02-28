import logging
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # 导入 CORS
from target_model import train_target_model,predict_target_model
from attack_model import perform_attack
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

# 初始化数据库表
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
        parameters TEXT
    )
    ''')
    conn.commit()
    conn.close()

# 在启动应用时初始化数据库
init_db()
# 静态文件配置
app = Flask(__name__, static_url_path="/static", static_folder="./data")
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
    data = request.json

    if not data or "target_label" not in data:
        return jsonify({"error": "Invalid request. 'target_label' is required."}), 400
    
    target_label = data["target_label"]
    
    # 验证 target_label
    if not isinstance(target_label, int) or target_label < 0 or target_label > 39:
        return jsonify({"error": "Invalid target_label. Must be an integer between 0 and 39."}), 400
    
    # 创建任务记录
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_db_connection()
    conn.execute('''
    INSERT INTO tasks (id, name, model, attack_type, status, progress, create_time, description, target_label)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        task_id,
        f"标签{target_label}的模型反演攻击",
        "目标分类器",
        "模型反演",
        'running',
        0,
        now,
        f"针对标签{target_label}的反演攻击",
        target_label
    ))
    conn.commit()
    conn.close()
    
    # 更新任务状态为运行中
    conn = get_db_connection()
    conn.execute('UPDATE tasks SET status = ?, start_time = ?, progress = ? WHERE id = ?', 
               ('running', now, 10, task_id))
    conn.commit()
    conn.close()
    
    try:
        # 执行攻击
        result_image = perform_attack(target_label)
        
        # 更新任务状态为完成
        conn = get_db_connection()
        conn.execute('UPDATE tasks SET status = ?, end_time = ?, progress = ? WHERE id = ?', 
                   ('completed', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100, task_id))
        conn.commit()
        conn.close()
        
        if result_image:
            return jsonify({"message": "Attack successful", "image": result_image, "task_id": task_id}), 200
        else:
            # 更新任务状态为失败
            conn = get_db_connection()
            conn.execute('UPDATE tasks SET status = ?, end_time = ?, error_message = ? WHERE id = ?', 
                       ('failed', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "攻击生成失败", task_id))
            conn.commit()
            conn.close()
            
            return jsonify({"error": "Attack failed", "task_id": task_id}), 500
    except Exception as e:
        # 更新任务状态为失败
        conn = get_db_connection()
        conn.execute('UPDATE tasks SET status = ?, end_time = ?, error_message = ? WHERE id = ?', 
                   ('failed', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(e), task_id))
        conn.commit()
        conn.close()
        
        return jsonify({"error": "Internal server error", "message": str(e), "task_id": task_id}), 500

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