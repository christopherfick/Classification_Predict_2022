from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score


class Performance():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = LogisticRegression(solver='liblinear', random_state=22)
        self.split_train_test()
        
    def split_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=22)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def f1_score(self):
        clf = self.model.fit(self.X_train, self.y_train)
        return f1_score(self.y_test, clf.predict(self.X_test), average='macro')
    
    def cross_val(self, scoring='f1_macro'):
        print(f'{scoring}:')
        return cross_val_score(self.model, self.X, self.y, cv=5, scoring=scoring)
    
    def model_report(self):
        clf = self.model.fit(self.X_train, self.y_train)
        print(classification_report(self.y_test, clf.predict(self.X_test)))