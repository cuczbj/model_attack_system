import os
import importlib
import json
import logging
import torch
import time
from typing import Dict, List, Tuple, Optional, Any
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路径配置
MODEL_DIR = "./models/classifiers"  # 模型定义目录
CHECKPOINT_DIR = "./checkpoint/target_model"  # 模型参数目录
MODEL_CONFIG_FILE = "./model_config.json"  # 模型配置文件

# 模型配置类型
class ModelConfig:
    def __init__(self, model_name: str, param_file: str, class_num: int = 0, 
                 input_shape: Tuple[int, int] = (0, 0), model_type: str = ""):
        self.model_name = model_name  # 模型定义文件名
        self.param_file = param_file  # 参数文件名
        self.class_num = class_num    # 分类数
        self.input_shape = input_shape  # 输入形状
        self.model_type = model_type    # 模型类型标识符
        self.created_time = time.time()  # 创建时间
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "param_file": self.param_file,
            "class_num": self.class_num,
            "input_shape": self.input_shape,
            "model_type": self.model_type,
            "created_time": self.created_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        config = cls(
            model_name=data.get("model_name", ""),
            param_file=data.get("param_file", ""),
            class_num=data.get("class_num", 0),
            input_shape=tuple(data.get("input_shape", (0, 0))),
            model_type=data.get("model_type", "")
        )
        config.created_time = data.get("created_time", time.time())
        return config


class ModelScanner:
    """模型扫描器，用于扫描模型定义和参数文件，并尝试自动匹配"""
    
    def __init__(self):
        self.model_defs = {}  # 模型定义 {name: class_name}
        self.param_files = []  # 参数文件列表
        self.model_configs = []  # 模型配置列表，包含匹配的定义和参数文件
        self.load_config()  # 加载已有配置
        
    def scan_model_definitions(self) -> Dict[str, str]:
        """扫描模型定义文件"""
        available_models = {}

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
            logger.warning(f"模型定义目录不存在，已创建: {MODEL_DIR}")
            return available_models

        for filename in os.listdir(MODEL_DIR):
            if filename.endswith(".py") and filename != "__init__.py" and filename != "evolve.py":
                module_name = filename[:-3]  # 去掉 `.py`
                try:
                    # 尝试导入模块
                    module = importlib.import_module(f'models.classifiers.{module_name}')
                    
                    # 读取模型类名称
                    if hasattr(module, "MODEL_CLASS_NAME"):
                        available_models[module_name] = getattr(module, "MODEL_CLASS_NAME")
                        logger.info(f"发现模型定义: {module_name} -> {getattr(module, 'MODEL_CLASS_NAME')}")
                    else:
                        logger.warning(f"模型 {module_name} 未定义 MODEL_CLASS_NAME")
                
                except Exception as e:
                    logger.error(f"加载模型 {module_name} 出错: {e}")

        return available_models
    
    def scan_param_files(self) -> List[str]:
        """扫描模型参数文件"""
        param_files = []
        
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            logger.warning(f"模型参数目录不存在，已创建: {CHECKPOINT_DIR}")
            return param_files
            
        for f in os.listdir(CHECKPOINT_DIR):
            if f.endswith((".pth", ".h5", ".tar", ".pkl")):
                param_files.append(f)
                logger.info(f"发现参数文件: {f}")
                
        return param_files
    
    def guess_model_type(self, param_file: str) -> Optional[str]:
        """根据参数文件名尝试猜测对应的模型类型"""
        # 尝试基于文件名的前缀匹配
        lower_name = param_file.lower()
        
        for model_name in self.model_defs.keys():
            if model_name.lower() in lower_name:
                return model_name
                
        # 通过简单启发式规则匹配
        if "mlp" in lower_name:
            return "MLP" if "MLP" in self.model_defs else None
        elif "vgg" in lower_name:
            return "VGG16" if "VGG16" in self.model_defs else None
        elif "facenet" in lower_name or "face64" in lower_name:
            return "FaceNet64" if "FaceNet64" in self.model_defs else None
        elif "ir152" in lower_name:
            return "IR152" if "IR152" in self.model_defs else None
            
        return None
    
    def try_match_model_and_param(self) -> List[ModelConfig]:
        """尝试自动匹配模型定义和参数文件"""
        new_configs = []
        
        # 检查参数文件是否已经在现有配置中
        existing_param_files = {config.param_file for config in self.model_configs}
        
        for param_file in self.param_files:
            # 跳过已配置的参数文件
            if param_file in existing_param_files:
                continue
                
            # 尝试猜测模型类型
            model_name = self.guess_model_type(param_file)
            if model_name:
                # 根据模型名称设置默认配置
                if model_name == "MLP":
                    config = ModelConfig(
                        model_name=model_name,
                        param_file=param_file,
                        class_num=40,  # AT&T Faces默认40类
                        input_shape=(112, 92),  # MLP默认输入形状
                        model_type="MLP"
                    )
                elif model_name in ["VGG16", "FaceNet64", "IR152"]:
                    config = ModelConfig(
                        model_name=model_name,
                        param_file=param_file,
                        class_num=1000,  # CelebA默认1000类
                        input_shape=(64, 64),  # 默认64x64输入
                        model_type=model_name
                    )
                else:
                    # 其他模型使用通用默认值
                    config = ModelConfig(
                        model_name=model_name,
                        param_file=param_file,
                        class_num=10,  # 默认10类
                        input_shape=(64, 64),  # 默认输入形状
                        model_type=model_name
                    )
                    
                new_configs.append(config)
                logger.info(f"自动匹配模型: {model_name} + {param_file}")
                
        return new_configs
    
    def validate_config(self, config: ModelConfig) -> bool:
        """验证模型配置是否有效"""
        # 检查模型定义是否存在
        if config.model_name not in self.model_defs:
            logger.error(f"模型定义不存在: {config.model_name}")
            return False
            
        # 检查参数文件是否存在
        if config.param_file not in self.param_files:
            logger.error(f"参数文件不存在: {config.param_file}")
            return False
            
        # 更多验证逻辑可以在这里添加
        return True
    
    def try_load_model(self, config: ModelConfig, device: str = "cpu") -> bool:
        """尝试加载模型以验证配置有效性"""
        try:
            # 动态导入模型文件
            model_module = importlib.import_module(f"models.classifiers.{config.model_name}")
            
            # 检查 MODEL_CLASS_NAME 是否存在
            if not hasattr(model_module, "MODEL_CLASS_NAME"):
                logger.error(f"MODEL_CLASS_NAME not found in {config.model_name}.py")
                return False

            model_class_name = getattr(model_module, "MODEL_CLASS_NAME")
            model_class = getattr(model_module, model_class_name, None)

            # 实例化模型
            model = model_class(config.class_num)
            
            # 如果有CUDA且模型支持，则使用GPU
            device_obj = torch.device(device)
            model = model.to(device_obj)

            # 加载模型参数
            param_file = os.path.join(CHECKPOINT_DIR, config.param_file)

            if param_file.endswith(".tar"):
                checkpoint = torch.load(param_file, map_location=device_obj)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(torch.load(param_file, map_location=device_obj))
                
            # 测试成功
            logger.info(f"模型加载测试成功: {config.model_name} + {config.param_file}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载测试失败: {e}")
            return False
    
    def scan(self) -> Tuple[Dict[str, str], List[str], List[ModelConfig]]:
        """执行完整扫描流程"""
        # 扫描模型定义和参数文件
        self.model_defs = self.scan_model_definitions()
        self.param_files = self.scan_param_files()
        
        # 尝试自动匹配并添加新配置
        new_configs = self.try_match_model_and_param()
        if new_configs:
            self.model_configs.extend(new_configs)
            self.save_config()
            
        return self.model_defs, self.param_files, self.model_configs
    
    def load_config(self) -> None:
        """从配置文件加载模型配置"""
        if os.path.exists(MODEL_CONFIG_FILE):
            try:
                with open(MODEL_CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                    
                self.model_configs = [ModelConfig.from_dict(item) for item in config_data]
                logger.info(f"已加载 {len(self.model_configs)} 个模型配置")
            except Exception as e:
                logger.error(f"加载模型配置出错: {e}")
                self.model_configs = []
        else:
            logger.info("模型配置文件不存在，将创建新配置")
            self.model_configs = []
    
    def save_config(self) -> None:
        """保存模型配置到文件"""
        try:
            config_data = [config.to_dict() for config in self.model_configs]
            with open(MODEL_CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"已保存 {len(self.model_configs)} 个模型配置")
        except Exception as e:
            logger.error(f"保存模型配置出错: {e}")
    
    def add_config(self, config: ModelConfig) -> bool:
        """添加新的模型配置"""
        # 验证配置
        if not self.validate_config(config):
            return False
            
        # 尝试加载模型测试配置是否有效
        if not self.try_load_model(config):
            return False
            
        # 检查是否存在同名配置
        for i, existing_config in enumerate(self.model_configs):
            if existing_config.model_name == config.model_name and existing_config.param_file == config.param_file:
                # 更新已有配置
                self.model_configs[i] = config
                self.save_config()
                logger.info(f"更新模型配置: {config.model_name} + {config.param_file}")
                return True
                
        # 添加新配置
        self.model_configs.append(config)
        self.save_config()
        logger.info(f"添加模型配置: {config.model_name} + {config.param_file}")
        return True
    
    def remove_config(self, model_name: str, param_file: str) -> bool:
        """移除模型配置"""
        for i, config in enumerate(self.model_configs):
            if config.model_name == model_name and config.param_file == param_file:
                self.model_configs.pop(i)
                self.save_config()
                logger.info(f"移除模型配置: {model_name} + {param_file}")
                return True
                
        logger.warning(f"未找到模型配置: {model_name} + {param_file}")
        return False


class ModelWatcher(FileSystemEventHandler):
    """文件系统监控器，监控模型文件变化"""
    
    def __init__(self, scanner: ModelScanner):
        self.scanner = scanner
        
    def on_created(self, event):
        """当文件被创建时触发"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        
        logger.info(f"检测到新文件: {file_path}")
        
        # 根据文件类型进行不同处理
        if file_path.startswith(MODEL_DIR) and file_name.endswith(".py"):
            # 新的模型定义文件
            logger.info(f"发现新模型定义: {file_name}")
            self.scanner.scan()
        elif file_path.startswith(CHECKPOINT_DIR) and file_name.endswith((".pth", ".h5", ".tar", ".pkl")):
            # 新的模型参数文件
            logger.info(f"发现新模型参数: {file_name}")
            self.scanner.scan()
            
    def on_deleted(self, event):
        """当文件被删除时触发"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        
        logger.info(f"检测到文件删除: {file_path}")
        
        # 更新模型列表和配置
        self.scanner.scan()
            
    def on_modified(self, event):
        """当文件被修改时触发"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        
        # 只关注模型定义文件的修改
        if file_path.startswith(MODEL_DIR) and file_name.endswith(".py"):
            logger.info(f"检测到模型定义修改: {file_path}")
            self.scanner.scan()


def start_model_watcher(scanner: ModelScanner) -> Observer:
    """启动文件系统监控器"""
    event_handler = ModelWatcher(scanner)
    observer = Observer()
    
    # 监控模型定义目录
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    observer.schedule(event_handler, MODEL_DIR, recursive=False)
    
    # 监控模型参数目录
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    observer.schedule(event_handler, CHECKPOINT_DIR, recursive=False)
    
    observer.start()
    logger.info("模型文件监控器已启动")
    
    return observer


# 单例模式获取模型扫描器
_scanner = None
def get_model_scanner() -> ModelScanner:
    global _scanner
    if _scanner is None:
        _scanner = ModelScanner()
        # 初始化时进行一次完整扫描
        _scanner.scan()
        
        # 在后台线程中启动监控器
        def run_watcher():
            observer = start_model_watcher(_scanner)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
            
        watcher_thread = Thread(target=run_watcher, daemon=True)
        watcher_thread.start()
        
    return _scanner


# 当直接运行此脚本时，执行模型扫描和监控
if __name__ == "__main__":
    scanner = get_model_scanner()
    model_defs, param_files, configs = scanner.scan()
    
    print(f"发现模型定义: {len(model_defs)}")
    for name, class_name in model_defs.items():
        print(f"  - {name}: {class_name}")
        
    print(f"发现参数文件: {len(param_files)}")
    for param_file in param_files:
        print(f"  - {param_file}")
        
    print(f"可用模型配置: {len(configs)}")
    for config in configs:
        print(f"  - {config.model_name} + {config.param_file} ({config.class_num}类)")
    
    # 启动监控
    print("启动文件监控，按Ctrl+C停止...")
    observer = start_model_watcher(scanner)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
