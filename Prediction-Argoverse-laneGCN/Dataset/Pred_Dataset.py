





def init(name, dataset_path):

    return Pred_Dataset(dataset_name=name, dataset_path=dataset_path)


class Pred_Dataset:
    def __init__(self, dataset_name, dataset_path=False):
        self.dataset_name = dataset_name
        print("加载的数据集为：", self.dataset_name)
        # 默认数据集路径
        if self.dataset_name == "argoverse":
            self.dataset_path = "./dataset/argoverse"
        if self.dataset_name == "argoverse2":
            self.dataset_path = "./dataset/argoverse2"
        if self.dataset_name == "interaction":
            self.dataset_path = "./dataset/interaction"

        if dataset_path:
            self.dataset_path = dataset_path
        print("默认数据集所在位置：", self.dataset_path)

        # 数据集默认参数
        if self.dataset_name == "Argoverse":
            self.hist_len = 20
            self.fut_len = 30
            self.with_map = True