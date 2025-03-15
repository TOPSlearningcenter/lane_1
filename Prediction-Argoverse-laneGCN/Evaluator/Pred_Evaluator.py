def evaluate(dataset, model="lanegcn", model_path="lanegcn_best.ckpt", metrics=['ADE', 'FDE', 'MR', 'DAC']):
    dataset.model = model
    dataset.model_path = model_path
    dataset.metrics = metrics
    print(f"————————————————{dataset.model}模型评估开始————————————————")
    from tdc_prediction_test import dataset_test
    dataset_test(dataset.model, dataset.model_path, metrics)
    print(f"————————————————{dataset.model}模型评估结束————————————————\n")

def visualize(dataset, model, model_path, vis_id, fig_path):
    dataset.model = model
    dataset.model_path = model_path
    dataset.viz_id = vis_id
    dataset.fig_path = fig_path
    from tdc_prediction_visualize import dataset_visualize
    dataset_visualize(dataset.model, dataset.model_path, dataset.viz_id, dataset.fig_path)


