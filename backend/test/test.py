# 结果可视化：用张博的方法将40张反转的图片放入目标模型的结果：
import os
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 配置参数
API_URL = "http://127.0.0.1:5000/predict"  # 替换为实际的预测接口地址
IMAGE_DIR = r"..\data\attack"  # 图片文件夹路径
OUTPUT_JSON = "./result/results.json"  # 保存结果的文件
VISUALIZATION_DIR = "result"  # 可视化结果保存路径

# 确保可视化目录存在
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def send_request(image_path):
    """发送图片到预测接口并返回响应结果"""
    with open(image_path, "rb") as f:
        files = {"image_file": f}
        response = requests.post(API_URL, files=files)
    response.raise_for_status()  # 检查请求是否成功
    return response.json()

def visualize_prediction(image_path, prediction, confidences, output_path):
    """可视化图片预测结果"""
    # 加载图片
    image = Image.open(image_path)

    # 找到置信度最高的类别及其置信度
    max_confidence = max(confidences)
    max_index = confidences.index(max_confidence)

    # 绘制图片和置信度分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title(f"Prediction: {prediction}")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(confidences)), confidences, color="blue")
    plt.title("Confidence Distribution")
    plt.xlabel("Class")
    plt.ylabel("Confidence")

    # 标记置信度最高的类别及其数值
    plt.text(max_index, max_confidence, f"{max_confidence:.2f}", 
             ha="center", va="bottom", color="red", fontsize=10)

    plt.tight_layout()

    # 保存可视化结果
    plt.savefig(output_path)
    plt.close()

def extract_label_from_filename(filename):
    """从文件名中提取真实标签"""
    return int(filename.split("_")[-1].split(".")[0])  # 提取最后一个字符作为标签

def main():
    # 结果数据
    results = []
    correct_count = 0
    total_count = 0

    # 遍历图片文件夹中的图片
    for file_name in os.listdir(IMAGE_DIR):
        if file_name.endswith(".png"):  # 只处理 PNG 图片
            image_path = os.path.join(IMAGE_DIR, file_name)
            try:
                # 提取真实标签
                true_label = extract_label_from_filename(file_name)

                # 发送预测请求
                response = send_request(image_path)

                # 提取返回值
                confidences = response.get("confidences", [])
                prediction = response.get("prediction", -1)

                # 保存结果
                results.append({
                    "image": file_name,
                    "true_label": true_label,
                    "prediction": prediction,
                    "confidences": confidences,
                })

                # 检查预测是否正确
                if prediction == true_label:
                    correct_count += 1
                total_count += 1

                # 可视化并保存
                visualization_path = os.path.join(VISUALIZATION_DIR, f"{file_name}_visualization.png")
                visualize_prediction(image_path, prediction, confidences, visualization_path)

                print(f"Processed {file_name}: Prediction={prediction}, True Label={true_label}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # 保存结果到 JSON 文件
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    # 输出正确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()