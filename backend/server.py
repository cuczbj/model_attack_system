import logging
from flask import Flask, request, jsonify
from target_model import train_target_model,predict_target_model
from attack_model import perform_attack
from PIL import Image
# 静态文件配置
app = Flask(__name__, static_url_path="/static", static_folder="./data")

# 训练目标模型,无调用，备用
@app.route("/train", methods=["POST"])
def train():
    response = train_target_model()
    return jsonify({"message": response})

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

# 反转攻击接口
@app.route("/attack", methods=["POST"])
def attack():
    data = request.json

    if not data or "target_label" not in data:
            return jsonify({"error": "Invalid request. 'target_label' is required."}), 400
    
    target_label = data["target_label"]  # 接收目标类别

    # 验证 target_label
    if not isinstance(target_label, int) or target_label < 0 or target_label > 39:
        return jsonify({"error": "Invalid target_label. Must be an integer between 0 and 39."}), 400
    
    logging.info(f"Received attack request for target_label: {target_label}")
    result_image_path = perform_attack(target_label)

    return jsonify({"message": "Attack successful", "result_image": result_image_path})
    # 构造完整的 URL 返回
    # server_url = request.host_url.rstrip("/")
    # full_image_url = f"{server_url}/{result_image_path}"

    # logging.info(f"Attack successful. Image saved at: {full_image_url}")

    # return jsonify({"message": "Attack successful", "result_image": full_image_url})
    # except Exception as e:
    # logging.error(f"Error during attack: {e}", exc_info=True)
    # return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)