from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
import io

# 攻击模型定义
class AttackModel(nn.Module):
    def __init__(self, output_features, input_features):
        super(AttackModel, self).__init__()
        self.fc = nn.Linear(output_features, input_features)

    def forward(self, x):
        return self.fc(x)

# 初始化攻击模型
input_features = 112 * 92
output_features = 40
attack_model = AttackModel(output_features, input_features)
attack_model.load_state_dict(torch.load("attack_model.pth"))
attack_model.eval()

# 数据预处理
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
target_model_url = "http://127.0.0.1:5000/predict"

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/attack", methods=["POST"])
def attack():
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("L")
    img_tensor = transform(image).unsqueeze(0)
    img_flatten = img_tensor.view(img_tensor.size(0), -1)

    # 调用目标模型
    response = requests.post(target_model_url, files={"file": file})
    prediction = response.json()["prediction"]
    pred_tensor = torch.tensor([prediction], dtype=torch.float32)

    # 使用攻击模型生成重构结果
    with torch.no_grad():
        reconstructed = attack_model(pred_tensor)
        reconstructed_image = reconstructed.view(1, 1, 112, 92).squeeze(0).numpy()

    # 将重构图像返回
    reconstructed_image = (reconstructed_image * 255).astype("uint8")
    img = Image.fromarray(reconstructed_image, mode="L")
    img_io = io.BytesIO()
    img.save(img_io, "PNG")
    img_io.seek(0)
    return app.response_class(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)