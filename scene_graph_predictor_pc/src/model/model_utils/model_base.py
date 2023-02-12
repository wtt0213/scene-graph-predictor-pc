import os
from tkinter import N
import torch
import torch.nn as nn
import collections

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.best_suffix = '_best.pth'
              
    def load(self, path):
        print('\nLoading %s model...' % self.name)
        loaded=True
        for name,model in self._modules.items():
            loaded &= self.loadWeights(model, os.path.join(path, name + self.best_suffix))

        if loaded:
            print('\tmodel loaded!\n')
        else:
            print('\tmodel loading failed!\n')
        return loaded

    def loadWeights(self, model, path):
        # print('isinstance(model, nn.DataParallel): ',isinstance(model, nn.DataParallel))
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
            new_dict = collections.OrderedDict()
            if isinstance(model, nn.DataParallel):
                for k,v in data['model'].items():
                    if k[:6] != 'module':
                        name = 'module.' + k
                        new_dict [name] = v
                model.load_state_dict(new_dict)
            else:
                for k,v in data['model'].items():
                    if k[:6] == 'module':
                        name = k[7:]
                        new_dict [name] = v
                model.load_state_dict(data['model'])
            return True
        else:
            return False