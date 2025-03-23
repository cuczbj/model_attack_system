from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import importlib
import os
import time

class ModelUploadHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.py') and event.src_path != '__init__.py':
            # 获取模块名称并输出
            module_name = os.path.basename(event.src_path)[:-3]
            # 替换模块名称中的非法字符（如 `-`）为下划线 `_`
            module_name = module_name.replace('-', '_')
            print(f"Attempting to import module: models.classifiers.{module_name}")
            try:
                # 动态导入模块
                importlib.import_module(f'.{module_name}', package='models.classifiers')
                print(f"Module {module_name} imported successfully!")
            except ModuleNotFoundError as e:
                print(f"Module {module_name} not found: {e}")
            except Exception as e:
                print(f"Error importing {module_name}: {e}")

def watch_and_load_models_with_watchdog(models_dir):
    event_handler = ModelUploadHandler()
    observer = Observer()
    observer.schedule(event_handler, models_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 启动监控
models_dir = './models/classifiers'
watch_and_load_models_with_watchdog(models_dir)