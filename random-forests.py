import math, random
from collections import Counter
class DecisionTree:
    def __init__(self,value=None):
        self.left = None
        self.right = None
        self.threshold = None
        self.feature = None
        self.value = None
    def set_feature(self, feature):
        self.feature = feature
    def set_threshold(self, threshold):
        self.threshold = threshold
    def set_value(self, value):
        self.value = value
    def set_left(self, tree):
        self.left = tree
        return self.left
    def set_right(self, tree):
        self.right = tree
        return self.right
    def decision(self, feature):
        if feature > self.threshold: #changed to greater than
            return self.right
        return self.left
    @property
    def is_leafy_green(self):
        return self.left == None and self.right == None
    def follow(self, features):
        curr = self.decision(features[self.feature])
        while not curr.is_leafy_green:
            curr = curr.decision(features[curr.feature])
        # if curr.value == None:
            # print 'no'
        return curr.value


f = open('hw12data/emailDataset/trainFeatures.csv', 'r')
f1 = open('hw12data/emailDataset/trainLabels.csv', 'r')
f2 = open('hw12data/emailDataset/valFeatures.csv', 'r')
f3 = open('hw12data/emailDataset/valLabels.csv', 'r')
train_features = f.read().splitlines()
train_labels = f1.read().splitlines()
train_labels = map(lambda x: int(x), train_labels)
train_features = [map(lambda y: float(y), x.split(',')) for x in train_features]
val_features = f2.read().splitlines()
val_features = [map(lambda y: float(y), x.split(',')) for x in val_features]
val_labels = f3.read().splitlines()
train_features_zipped = zip(train_features, train_labels)

def entropy(labels):
    impure1 = float(sum(labels))/len(labels)
    impure2 = float(len(labels)-sum(labels))/len(labels)
    temp1 = 0
    temp2 = 0
    if impure1 != 0:
        temp1 = impure1*math.log(impure1, 2)
    if impure2 != 0:
        temp2 = impure2*math.log(impure2, 2)
    return -(temp1+temp2)

def best_feature(training_data, features):
    labels = [x[1] for x in training_data]
    examples = [x[0] for x in training_data]
    best_feature = 0
    best_goodness = -float("inf")
    best_threshold = 0
    for feature_idx in features:
        sorted_feature = sorted(list(set([x[feature_idx] for x in examples])))
        if len(sorted_feature) == 1:
            return feature_idx, sorted_feature[0]
        threshold, goodness = find_me_a_threshold(sorted_feature, training_data, feature_idx)
        if goodness > best_goodness:
            best_feature = feature_idx
            best_goodness = goodness
            best_threshold = threshold
    return best_feature, best_threshold #changed it to best_threshold

def find_me_a_threshold(sorted_feature, training_data, feature): #what if everything is all 1s or 0s; len(sorted_feature) == 1?
    best_goodness = -float("inf")
    best_threshold = 0
    labels = [x[1] for x in training_data]
    curr_entropy = entropy(labels)
    for i in range(0,len(sorted_feature)):
        if i == len(sorted_feature)-1:
            float(sorted_feature[i])
        else:
            threshold = float(sorted_feature[i] + sorted_feature[i+1])/2.0
        left = []
        right = []
        for feature_vector, label in training_data:
            if feature_vector[feature] <= threshold:
                left.append(label)
            else:
                right.append(label)
        goodness = curr_entropy - ((len(left)/len(training_data))*entropy(left) + (len(right)/len(training_data))*entropy(right)) #length of the sorted features instead of training_data
        if goodness > best_goodness:
            best_goodness = goodness
            best_threshold = threshold
    if best_goodness == -float("inf"):
        print "ASFHIDFJasidhfasjfaskjhgdfaowefbask"
    return best_threshold, best_goodness 

def make_decision_tree(training_data, bagged_values):
    random.shuffle(bagged_values)
    feature_samples = bagged_values[0:8]
    tree = DecisionTree()
    training_data_length = len(training_data)
    labels = [example[1] for example in training_data]
    if sum(labels) == training_data_length:
        tree.set_value(1)
        return tree
    elif sum(labels) == 0:
        tree.set_value(0)
        return tree
    if len(bagged_values) <= 1:
        if sum(labels) > training_data_length/2.0:
            tree.set_value(1)
            return tree
        else:
            tree.set_value(0)
            return tree
    feature, threshold = best_feature(training_data, feature_samples)
    tree.set_threshold(threshold)
    tree.set_feature(feature)
    bagged_values.remove(feature)
    left_training_data = [x for x in training_data if x[0][feature] <= threshold]
    left_training_labels = [x[1] for x in training_data if x[0][feature] <= threshold]
    right_training_data = [x for x in training_data if x[0][feature] > threshold]
    right_training_labels = [x[1] for x in training_data if x[0][feature] > threshold]
    if len(left_training_data) == 0:
        if sum(right_training_labels) > len(right_training_labels)/2.0:
            tree.left = DecisionTree(0)
        else:
            tree.left = DecisionTree(1)
        tree.right = make_decision_tree(right_training_data, bagged_values)
    elif len(right_training_data) == 0:
        if sum(left_training_labels) > len(left_training_labels)/2.0:
            tree.right = DecisionTree(0)
        else:
            tree.right = DecisionTree(1)
        tree.left = make_decision_tree(left_training_data, bagged_values)
    else:
        tree.left = make_decision_tree(left_training_data, bagged_values)
        tree.right = make_decision_tree(right_training_data, bagged_values)
    return tree

T = [1, 2, 5, 10, 25]
trees = []
temp = range(57)
for t in [2]:
    while len(trees) < t:
        bagged_values = [] #making a random tree every time
        for m in range(0,57):
            bagged_values.append(random.choice(temp)) #bagging values, get rid of them
        print "Making a tree"
        trees.append(make_decision_tree(train_features_zipped, bagged_values))
    count = 0
    for i in range(len(val_features)):
        tally = [tree.follow(val_features[i]) for tree in trees]
        result = Counter(tally).most_common(1)[0][0]
        if result != int(val_labels[i]):
            print result
            count += 1
    print "you fucked up  with T of", str(t), "and a percentage of ", float(count)/len(val_features)


