# 数据集定义
from Dataset import Pred_Dataset
dataset = Pred_Dataset.init(name="Argoverse", dataset_path='./dataset')


# 数据预处理
from Processor import Pred_Processor
Pred_Processor.preprocess(dataset, hist_len=20, fut_len=30, output_path='./dataset/preprocess')


# 模型调用与训练
from Example_model import Pred_model
Pred_model.train(dataset, model='LaneGCN', save_path="./results/LaneGCN", epochs=5)


# 模型评估&可视化
from Evaluator import Pred_Evaluator
Pred_Evaluator.evaluate(dataset, model='LaneGCN', model_path="lanegcn_best.ckpt", metrics=['ADE','FDE','MR','DAC'])

Pred_Evaluator.visualize(dataset, model="LaneGCN", model_path="lanegcn_best.ckpt", vis_id=78, fig_path="./78.jpg")

