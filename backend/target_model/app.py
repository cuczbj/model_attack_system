from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# 模型初始化
class_num = 40
h, w = 112, 92
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义目标模型结构
class MLP(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(MLP, self).__init__()
        self.fc = torch.nn.Linear(input_features, 3000)
        self.regression = torch.nn.Linear(3000, output_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.regression(x)
        return x

# 加载模型
model = MLP(h * w, class_num).to(device)
model.load_state_dict(torch.load("mynet_50.pkl"))
model.eval()

# 数据预处理
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # print(request.files)  # 输出调试信息
    # file = request.files["file"]
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    image = Image.open(io.BytesIO(file.read())).convert("L")
    img_tensor = transform(image).unsqueeze(0).to(device)  # 转换为批次张量
    img_flatten = img_tensor.view(img_tensor.size(0), -1)
    # with torch.no_grad():
    #     output = model(img_flatten)
    #     prediction = torch.argmax(output, dim=1).item()
    # return jsonify({"prediction": prediction})
    with torch.no_grad():
        output = model(img_flatten)  # 模型输出
        probabilities = torch.nn.functional.softmax(output, dim=1)  # 计算概率
        confidence, prediction = torch.max(probabilities, dim=1)  # 获取置信度和预测类别
    return jsonify({
        "prediction": prediction.item(),
        "confidence": confidence.item(),
        "probabilities": probabilities.squeeze().tolist()  # 返回每一类别的概率
    
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
