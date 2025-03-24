from PIL import Image

def convert_to_grayscale(image_path, save_path):
    image = Image.open(image_path).convert("L")  # 转换为灰度
    image.save(save_path)  # 保存灰度图像
    print(f"灰度图像已保存至: {save_path}")

# 示例使用
image_path = "test_MLP_ATT.png"  # 替换为你的图像路径
save_path = "test_MLP_ATT1.png"  # 目标灰度图像保存路径
convert_to_grayscale(image_path, save_path)