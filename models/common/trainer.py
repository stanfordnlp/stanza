import torch

class Trainer:
    def change_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def save(self, filename):
        savedict = {
                   'model': self.model.state_dict(),
                   'optimizer': self.optimizer.state_dict()
                   }
        torch.save(savedict, filename)

    def load(self, filename):
        savedict = torch.load(filename, lambda storage, loc: storage)

        self.model.load_state_dict(savedict['model'])
        if self.args['mode'] == 'train':
            self.optimizer.load_state_dict(savedict['optimizer'])
