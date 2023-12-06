import torch
import os


def save_models(models, save_path):
    # 创建文件夹（如果不存在）
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 如果传入的模型是单个模型，将其放入列表中以便后续统一处理
    if not isinstance(models, list):
        models = [models]

    # 保存每个接收到的模型
    for idx, model in enumerate(models):
        model_path = os.path.join(save_path, f"model_{idx}.pt")  # 模型保存路径，可根据需要修改文件名
        torch.save(model.parameters.state_dict(), model_path)
