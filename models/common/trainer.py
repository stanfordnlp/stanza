import pickle

class Trainer:
    def change_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def save(self, filename):
        savedict = {
                   'model': self.model.state_dict(),
                   'optimizer': self.optimizer.state_dict()
                   }
        with open(filename, 'wb') as f:
            pickle.dump(savedict, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            savedict = pickle.load(f)

        self.model.load_state_dict(savedict['model'])
        if self.args['mode'] == 'train':
            self.optimizer.load_state_dict(savedict['optimizer'])
