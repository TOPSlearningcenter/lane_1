

def preprocess(dataset, hist_len, fut_len, output_path):
    dataset.hist_len = hist_len
    dataset.fut_len = fut_len
    dataset.output_path = output_path
    print(f"————————————————{dataset.dataset_name}数据集预处理开始————————————————")
    from tdc_prediction_preprocess import preprocess_dataset
    preprocess_dataset(dataset.dataset_name, dataset.dataset_path, dataset.hist_len, dataset.fut_len, dataset.output_path)
    print(f"————————————————{dataset.dataset_name}数据集预处理结束————————————————\n")



