class MOEW(object):
    def __init__(self, model, pred_type=None):
        self.model = model
        self.pred_type = pred_type

    def fit(self, train_X, valid_X, train_y, valid_y):
        self._check_type(train_y)

    def predict(self, test_X):
        pass

    def _check_type(self, train_y):
        if not self.pred_type:
            if train_y.dtype in ['category', 'object']:
                self.pred_type = 'cls'
            else:
                self.pred_type = 'reg'

        if self.pred_type == 'cls':
            self.num_classes = train_y.nunique()
        elif self.pred_type == 'reg':
            self.num_classes = 1
        else:
            raise ValueError(f'Invalid pred_type {self.pred_type}, should be `reg` or `cls`')
