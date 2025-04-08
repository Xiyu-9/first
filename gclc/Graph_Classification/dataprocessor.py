import numpy as np

class DataProcessor:
    def __init__(self, data):
        """
        初始化数据处理器
        :param data: 包含 'features'、'adj_list' 和 'targets' 的字典
        """
        self.data = data

    def calculate_category_avg(self):
        """
        计算每个类别的 avg（最大和最小 feature 长度的平均值）
        :return: 每个类别的 avg 字典
        """
        categories = np.unique(self.data['targets'])
        category_avgs = {}

        for category in categories:
            # 获取当前类别的索引
            indices = [i for i, target in enumerate(self.data['targets']) if target == category]

            # 获取当前类别的 features 长度
            feature_lengths = [len(self.data['features'][i]) for i in indices]

            # 找到最大和最小长度
            max_length = max(feature_lengths)
            min_length = min(feature_lengths)

            # 计算平均值（取整）
            avg_length = (max_length + min_length) // 2
            category_avgs[category] = avg_length

        return category_avgs

    def filter_data_by_avg(self, category_avgs):
        """
        根据每个类别的 avg 删除不符合条件的数据
        :param category_avgs: 每个类别的 avg 字典
        """
        # 初始化保留的索引
        indices_to_keep = []

        # 存储增强后的样本
        processed_data = {
            'features': [],
            'adj_list': [],
            'targets': []
        }
        for category, avg in category_avgs.items():
            # 获取当前类别的索引
            indices = [i for i, target in enumerate(self.data['targets']) if target == category]

            # 筛选出元素个数大于等于 avg 的索引
            valid_indices = [i for i in indices if len(self.data['features'][i]) >= avg]
            indices_to_keep.extend(valid_indices)

        # 按索引保留数据
        self.data['features'] = [self.data['features'][i] for i in indices_to_keep]
        self.data['adj_list'] = [self.data['adj_list'][i] for i in indices_to_keep]
        self.data['targets'] = np.array(self.data['targets'])[np.array(indices_to_keep)]

        #self.data['targets'] = self.data['targets'][np.array(indices_to_keep)]  # 转换为 NumPy 数组进行索引

    def process(self):
        """
        执行数据处理流程：计算 avg 并过滤数据
        """
        category_avgs = self.calculate_category_avg()
        self.filter_data_by_avg(category_avgs)
