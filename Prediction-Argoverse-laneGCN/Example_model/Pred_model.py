def train(dataset, model, save_path, epochs=5):
    dataset.model = model
    dataset.epochs = epochs
    dataset.save_path = save_path

    print(f"————————————————{dataset.model}模型训练开始————————————————")
    from tdc_prediction_train import dataset_train
    dataset_train(dataset.model, dataset.epochs, dataset.save_path)
    print(f"————————————————{dataset.model}模型训练结束————————————————\n")