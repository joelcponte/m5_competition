import pandas as pd
import numpy as np
import warnings

class M5TimeSeriesSplit:

    def __init__(self, n_days, days_columns, fixed_columns=None, 
                 sliding_window=True, return_index=True,
                 rename=False, method=1, split_train=True,
                 do_enumerate=False):
        
        self.n_days = n_days
        self.days_columns = days_columns
        self.fixed_columns = fixed_columns
        self.sliding_window = sliding_window
        self.return_index = return_index
        self.rename = rename
        self.split_train = split_train
        self.method = method
        self.enumerate = do_enumerate
        
        assert isinstance(days_columns, list)
        assert isinstance(fixed_columns, list)

    def split(self, X, y=None):

        for i, day in enumerate(range(self.n_days)):
            X_train, y_train, X_test, y_test = self.create_split(day, X, y)
            
            if self.rename:
                y_test.columns = [f"F{day+1}"]
                
            if self.enumerate:
                yield i, X_train, y_train, X_test, y_test
            else:
                yield X_train, y_train, X_test, y_test

    def create_split(self, day, X, y=None):
        
        if self.method == 1:
            return self._split_1(day, X, y)
        elif self.method == 2:
            return self._split_2(day, X, y)
        elif self.method == 3:
            return self._split_3(day, X, y)
        else:
            raise Exception("Only methods 1, 2 or 3 are allowed.")

    
    def _split_1(self, day, X, y=None):
        
    
        if self.sliding_window:
            te_day0 = day + 1
        else: 
            te_day0 = 0
        
        if self.split_train:
            tr_x = self.fixed_columns + self.days_columns[:-(self.n_days + day + 1)]
            tr_y = [self.days_columns[-(self.n_days + 1)]]
        else:
            tr_x = self.fixed_columns + self.days_columns[:-(self.n_days + 1)]
            tr_y = []
        te_x = self.fixed_columns + self.days_columns[te_day0:-(self.n_days)]
        te_y = [self.days_columns[-(self.n_days - day)]]
        
        
        if self.return_index:
            return tr_x, tr_y, te_x, te_y
        
        return X[tr_x], X[tr_y], X[te_x], X[te_y]
    
    def _split_2(self, day, X, y=None):
        
        if not self.split_train:
            warnings.warn("split_train=False ignored for method 2.")
            
        if self.sliding_window:
            tr_day0 = day
            te_day0 = day + 1
        else: 
            tr_day0 = 0
            te_day0 = 0
        
        
        tr_x = self.fixed_columns + self.days_columns[tr_day0:-(2*day + 2)]
        tr_y = [self.days_columns[-(2+day)]]
        te_x = self.fixed_columns + self.days_columns[te_day0:-(day + 1)]
        te_y = [self.days_columns[-1]]
        
        
        if self.return_index:
            return tr_x, tr_y, te_x, te_y
        
        return X[tr_x], X[tr_y], X[te_x], X[te_y]
    
    def _split_3(self, day, X, y=None):
        
        if not self.split_train:
            warnings.warn("split_train=False ignored for method 3.")

        if self.sliding_window:
            tr_day0 = day
            te_day0 = day + 1
        else: 
            tr_day0 = 0
            te_day0 = 0
        
        
        tr_x = self.fixed_columns + self.days_columns[tr_day0:-(self.n_days + 1)]
        tr_y = [self.days_columns[-(self.n_days - day + 1)]]
        te_x = self.fixed_columns + self.days_columns[te_day0:-(self.n_days)]
        te_y = [self.days_columns[-(self.n_days - day)]]
        
        
        if self.return_index:
            return tr_x, tr_y, te_x, te_y
        
        return X[tr_x], X[tr_y], X[te_x], X[te_y]
