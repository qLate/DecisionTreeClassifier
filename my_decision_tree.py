from sklearn.model_selection import train_test_split


class Node:

    def __init__(self, feature_index, threashold, left, right, value=None):
        self.feature_index = feature_index
        self.threshold = threashold
        self.left = left
        self.right = right
        self.value = value


class MyDecisionTreeClassifier:

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def gini_single(self, group, classes):
        sum = 0
        for cls in classes:
            sum += (group.count(cls)/len(group))**2
        return 1-sum

    def get_info_gain(self, groups, classes):

        samples_count = sum([len(group) for group in groups])

        gini_sum = self.gini_single(groups[0]+groups[1], classes)
        for group in groups:
            if len(group) == 0:
                continue

            koef = len(group)/samples_count
            gini_sum -= self.gini_single(group, classes)*koef
        return gini_sum

    def split_data(self, X, y):
        best_split = None
        best_split_gain = -float('inf')

        classes = list(set(y))

        for feature_index in range(len(X[0])):
            possible_threasholds = []

            data = [x+[y_] for x, y_ in zip(X, y)]
            data = sorted(data, key=lambda sample: sample[feature_index])
            X = [sample[:-1] for sample in data]
            y = [sample[-1] for sample in data]
            for sample in X:
                possible_threasholds.append(sample[feature_index])
            for value_index, threashold in enumerate(possible_threasholds):
                group1 = y[:value_index]
                group2 = y[value_index:]
                if len(group1) > 0 and len(group2) > 0:
                    new_gain = self.get_info_gain([group1, group2], classes)
                    if best_split_gain < new_gain:
                        best_split_gain = new_gain
                        best_split = (
                            feature_index, threashold, X[:value_index], X[value_index:], group1, group2)
        return best_split

    def build_tree(self, X, y, depth=0):

        if len(set(y)) > 1 and depth < self.max_depth:
            feature_index, threashold, X1, X2, y1, y2 = self.split_data(X, y)
            left_child = self.build_tree(X1, y1, depth+1)
            right_child = self.build_tree(X2, y2, depth+1)
            return Node(feature_index, threashold, left_child, right_child, None)

        return Node(-1, float('inf'), None, None, max(y, key=y.count))

    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)

    def predict(self, X_test):
        predictions = [self.make_predict(x, self.root) for x in X_test]
        return predictions

    def make_predict(self, x, tree):
        if tree.value != None:
            return tree.value
        
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_predict(x, tree.left)
        return self.make_predict(x, tree.right)


y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
X = [[4.9, 3., 1.4, 0.2],
     [4.7, 3.2, 1.3, 0.2],
     [4.6, 3.1, 1.5, 0.2],
     [5., 3.6, 1.4, 0.2],
     [5.4, 3.9, 1.7, 0.4],
     [4.6, 3.4, 1.4, 0.3],
     [5., 3.4, 1.5, 0.2],
     [4.4, 2.9, 1.4, 0.2],
     [4.9, 3.1, 1.5, 0.1],
     [5.4, 3.7, 1.5, 0.2],
     [4.8, 3.4, 1.6, 0.2],
     [4.8, 3., 1.4, 0.1],
     [4.3, 3., 1.1, 0.1],
     [5.8, 4., 1.2, 0.2],
     [5.7, 4.4, 1.5, 0.4],
     [5.4, 3.9, 1.3, 0.4],
     [5.1, 3.5, 1.4, 0.3],
     [5.7, 3.8, 1.7, 0.3],
     [5.1, 3.8, 1.5, 0.3],
     [5.4, 3.4, 1.7, 0.2],
     [5.1, 3.7, 1.5, 0.4],
     [4.6, 3.6, 1., 0.2],
     [5.1, 3.3, 1.7, 0.5],
     [4.8, 3.4, 1.9, 0.2],
     [5., 3., 1.6, 0.2],
     [5., 3.4, 1.6, 0.4],
     [5.2, 3.5, 1.5, 0.2],
     [5.2, 3.4, 1.4, 0.2],
     [4.7, 3.2, 1.6, 0.2],
     [4.8, 3.1, 1.6, 0.2],
     [5.4, 3.4, 1.5, 0.4],
     [5.2, 4.1, 1.5, 0.1],
     [5.5, 4.2, 1.4, 0.2],
     [4.9, 3.1, 1.5, 0.2],
     [5., 3.2, 1.2, 0.2],
     [5.5, 3.5, 1.3, 0.2],
     [4.9, 3.6, 1.4, 0.1],
     [4.4, 3., 1.3, 0.2],
     [5.1, 3.4, 1.5, 0.2],
     [5., 3.5, 1.3, 0.3],
     [4.5, 2.3, 1.3, 0.3],
     [4.4, 3.2, 1.3, 0.2],
     [5., 3.5, 1.6, 0.6],
     [5.1, 3.8, 1.9, 0.4],
     [4.8, 3., 1.4, 0.3],
     [5.1, 3.8, 1.6, 0.2],
     [4.6, 3.2, 1.4, 0.2],
     [5.3, 3.7, 1.5, 0.2],
     [5., 3.3, 1.4, 0.2],
     [7., 3.2, 4.7, 1.4],
     [6.4, 3.2, 4.5, 1.5],
     [6.9, 3.1, 4.9, 1.5],
     [5.5, 2.3, 4., 1.3],
     [6.5, 2.8, 4.6, 1.5],
     [5.7, 2.8, 4.5, 1.3],
     [6.3, 3.3, 4.7, 1.6],
     [4.9, 2.4, 3.3, 1.],
     [6.6, 2.9, 4.6, 1.3],
     [5.2, 2.7, 3.9, 1.4],
     [5., 2., 3.5, 1.],
     [5.9, 3., 4.2, 1.5],
     [6., 2.2, 4., 1.],
     [6.1, 2.9, 4.7, 1.4],
     [5.6, 2.9, 3.6, 1.3],
     [6.7, 3.1, 4.4, 1.4],
     [5.6, 3., 4.5, 1.5],
     [5.8, 2.7, 4.1, 1.],
     [6.2, 2.2, 4.5, 1.5],
     [5.6, 2.5, 3.9, 1.1],
     [5.9, 3.2, 4.8, 1.8],
     [6.1, 2.8, 4., 1.3],
     [6.3, 2.5, 4.9, 1.5],
     [6.1, 2.8, 4.7, 1.2],
     [6.4, 2.9, 4.3, 1.3],
     [6.6, 3., 4.4, 1.4],
     [6.8, 2.8, 4.8, 1.4],
     [6.7, 3., 5., 1.7],
     [6., 2.9, 4.5, 1.5],
     [5.7, 2.6, 3.5, 1.],
     [5.5, 2.4, 3.8, 1.1],
     [5.5, 2.4, 3.7, 1.],
     [5.8, 2.7, 3.9, 1.2],
     [6., 2.7, 5.1, 1.6],
     [5.4, 3., 4.5, 1.5],
     [6., 3.4, 4.5, 1.6],
     [6.7, 3.1, 4.7, 1.5],
     [6.3, 2.3, 4.4, 1.3],
     [5.6, 3., 4.1, 1.3],
     [5.5, 2.5, 4., 1.3],
     [5.5, 2.6, 4.4, 1.2],
     [6.1, 3., 4.6, 1.4],
     [5.8, 2.6, 4., 1.2],
     [5., 2.3, 3.3, 1.],
     [5.6, 2.7, 4.2, 1.3],
     [5.7, 3., 4.2, 1.2],
     [5.7, 2.9, 4.2, 1.3],
     [6.2, 2.9, 4.3, 1.3],
     [5.1, 2.5, 3., 1.1],
     [5.7, 2.8, 4.1, 1.3],
     [6.3, 3.3, 6., 2.5],
     [5.8, 2.7, 5.1, 1.9],
     [7.1, 3., 5.9, 2.1],
     [6.3, 2.9, 5.6, 1.8],
     [6.5, 3., 5.8, 2.2],
     [7.6, 3., 6.6, 2.1],
     [4.9, 2.5, 4.5, 1.7],
     [7.3, 2.9, 6.3, 1.8],
     [6.7, 2.5, 5.8, 1.8],
     [7.2, 3.6, 6.1, 2.5],
     [6.5, 3.2, 5.1, 2.],
     [6.4, 2.7, 5.3, 1.9],
     [6.8, 3., 5.5, 2.1],
     [5.7, 2.5, 5., 2.],
     [5.8, 2.8, 5.1, 2.4],
     [6.4, 3.2, 5.3, 2.3],
     [6.5, 3., 5.5, 1.8],
     [7.7, 3.8, 6.7, 2.2],
     [7.7, 2.6, 6.9, 2.3],
     [6., 2.2, 5., 1.5],
     [6.9, 3.2, 5.7, 2.3],
     [5.6, 2.8, 4.9, 2.],
     [7.7, 2.8, 6.7, 2.],
     [6.3, 2.7, 4.9, 1.8],
     [6.7, 3.3, 5.7, 2.1],
     [7.2, 3.2, 6., 1.8],
     [6.2, 2.8, 4.8, 1.8],
     [6.1, 3., 4.9, 1.8],
     [6.4, 2.8, 5.6, 2.1],
     [7.2, 3., 5.8, 1.6],
     [7.4, 2.8, 6.1, 1.9],
     [7.9, 3.8, 6.4, 2.],
     [6.4, 2.8, 5.6, 2.2],
     [6.3, 2.8, 5.1, 1.5],
     [6.1, 2.6, 5.6, 1.4],
     [7.7, 3., 6.1, 2.3],
     [6.3, 3.4, 5.6, 2.4],
     [6.4, 3.1, 5.5, 1.8],
     [6., 3., 4.8, 1.8],
     [6.9, 3.1, 5.4, 2.1],
     [6.7, 3.1, 5.6, 2.4],
     [6.9, 3.1, 5.1, 2.3],
     [5.8, 2.7, 5.1, 1.9],
     [6.8, 3.2, 5.9, 2.3],
     [6.7, 3.3, 5.7, 2.5],
     [6.7, 3., 5.2, 2.3],
     [6.3, 2.5, 5., 1.9],
     [6.5, 3., 5.2, 2.],
     [6.2, 3.4, 5.4, 2.3],
     ]
X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)

clf = MyDecisionTreeClassifier(100)
clf.fit(X, y)

predictions = clf.predict(X_test)
right_predictions = [predictions[index] for index in range(
    len(predictions)) if predictions[index] == y_test[index]]

print(predictions)
print(len(right_predictions)/len(predictions))
