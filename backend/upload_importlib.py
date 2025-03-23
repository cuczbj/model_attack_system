import importlib
import os
import torch

MODEL_DIR = "./models/classifiers"
CHECKPOINT_DIR = "./checkpoint/target_model"

# 获取所有可用模型
def get_available_models():
    """ 获取所有可用的模型（即 `models.classifiers` 目录下的 Python 文件）"""
    available_models = {}

    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".py") and filename != "__init__.py" and filename!="evolve.py":
            module_name = filename[:-3]  # 去掉 `.py`
            try:
                module = importlib.import_module(f'models.classifiers.{module_name}')
                
                # 读取模型类名称
                if hasattr(module, "MODEL_CLASS_NAME"):
                    available_models[module_name] = getattr(module, "MODEL_CLASS_NAME")
                else:
                    print(f"Warning: {module_name} does not define MODEL_CLASS_NAME")
            
            except Exception as e:
                print(f"Error loading {module_name}: {e}")

    return available_models

# 获取所有可用模型的参数
def get_available_params():
    """ 获取所有可用的模型参数文件 """
    return [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith((".pth", ".h5",".tar",".pkl"))]


#目前先只支持需要输入输出维度的模型，后续可以增加模型参数的自动获取
# def load_model(model_name, param_filename, h , w ,class_num) :
#     try:
#         # 动态导入模型文件
#         model_module = importlib.import_module(f"models.classifiers.{model_name}")
#         # 检查 MODEL_CLASS_NAME 是否存在
#         if not hasattr(model_module, "MODEL_CLASS_NAME"):
#             raise RuntimeError(f"MODEL_CLASS_NAME not found in {model_name}.py")

#         model_class_name = getattr(model_module, "MODEL_CLASS_NAME")  # 变量是字符串
#         model_class = getattr(model_module, model_class_name, None)  # 获取类

#         # 计算输入和输出维度
#         input_features = h * w
#         output_features = class_num

#         # 实例化模型
#         model = model_class(input_features, output_features)

#         # 加载模型参数
#         param_file = os.path.join(CHECKPOINT_DIR, param_filename)
#         model = model_class()
#         model.load_state_dict(torch.load(param_file))  # 加载权重
#         model.eval()  # 切换到评估模式
#         return model
#     except Exception as e:
#         raise RuntimeError(f"Error loading model {model_name}: {str(e)}")

def load_model(model_name, param_filename, h , w ,class_num) :
    try:
        # 动态导入模型文件
        model_module = importlib.import_module(f"models.classifiers.{model_name}")
        
        # 检查 MODEL_CLASS_NAME 是否存在
        if not hasattr(model_module, "MODEL_CLASS_NAME"):
            raise RuntimeError(f"MODEL_CLASS_NAME not found in {model_name}.py")

        model_class_name = getattr(model_module, "MODEL_CLASS_NAME")  # 变量是字符串
        model_class = getattr(model_module, model_class_name, None)  # 获取类

        # 计算输入和输出维度
        input_features = h * w
        output_features = class_num

        # 实例化模型（传入计算得到的维度）
        model = model_class(input_features, output_features)

        # 加载模型参数
        # param_file = os.path.join(CHECKPOINT_DIR, param_filename)
        # 直接拼接路径
        param_file = CHECKPOINT_DIR + "/" + param_filename
        model.load_state_dict(torch.load(param_file))  # 加载权重
        model.eval()  # 切换到评估模式
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {str(e)}")




# 示例：获取所有可用模型
# available_models = get_available_models()
# print("Available models:", available_models)
def run_inference(model_instance, input_data):
    """ 运行推理 """
    if model_instance and hasattr(model_instance, "predict"):
        result = model_instance.predict(input_data)
        print("Inference result:", result)
        return result
    else:
        print("Model instance is invalid or lacks 'predict' method.")
        return None