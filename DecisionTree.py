import numpy as np

class _Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.features = None
        self.threshold = None
        self.label = None
        self.numdata = None
        self.gini_index = None

    def build(self, data, target):
        self.numdata = data.shape[0]
        num_features = data.shape[1]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        class_cnt = {i: len(target[target == i]) for i in np.unique(target)}
        self.label = max(class_cnt.itemos(), key = lambda x: x[1])[0]

        best_gini_index = 0.0
        best_feature = None
        best_threshold = None

        gini = self.gini_func(target)

        for f in range(num_features):
            data_f = np.unique(data[:, f])
            points = (data_f[:, -1] + data_f[1:]) / 2.0

            for threshold in points:
                target_l = target[data[:, f] < threshold]
                target_r = target[data[:, f] >= threshold]

                gini_l = self.gini_func(target_l)
                gini_r = self.gini_func(target_r)
                pl = float(target_l.shape[0]) / self.numdata
                pr = float(target_r.shape[0]) / self.numdata
                gini_index = gini - (pl * gini_l + pr * gini_r)

                if gini_index > best_gini_index:
                    best_gini_index = gini_index
                    best_feature = f
                    best_threshold = threshold

        if best_gini_index == 0:
            return

        self.feature = best_feature
        self.gini_index = best_gini_index
        self.threshold = best_threshold

        data_l = data[data[:, self.feature] < self.threshold]
        target_l = target[data[:, self.feature] < self.threshold]
        self.right = _Node()
        self.right.build(data_r, target_r)
