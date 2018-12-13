"""
Utilities for managing parameters.
"""
import json
import os

def load_param(filename):
    with open(filename) as infile:
        p = json.load(infile)
    return p

def save_param(param, filename):
    with open(filename, 'w') as outfile:
        json.dump(param, outfile, indent=2)
    return

class ParamManager():
    def __init__(self, param_dir, lang):
        """ Initialize a param manager with file dirs. """
        if not os.path.exists(param_dir):
            raise Exception("Cannot find params dir at: " + param_dir)
        self.param_dir = param_dir
        self.lang = lang
        # check default
        self.default_file = self.param_dir + '/default.json'
        if not os.path.exists(self.default_file):
            raise Exception("Cannot find default parameter file at: " + self.default_file)
        self.param_file = self.param_dir + '/{}_params.json'.format(lang)
        
    @property
    def params(self):
        if not hasattr(self, '_params'):
            if not os.path.exists(self.param_file):
                # load default and save to language-specific file
                self._params = load_param(self.default_file)
                save_param(self._params, self.param_file)
            else:
                self._params = load_param(self.param_file)
        assert "best_dev_score" in self._params, "Cannot find best_dev_score in saved parameters."
        return self._params

    @property
    def param_names(self):
        """ Return list of parameter names. """
        if not hasattr(self, '_param_names'):
            self._param_names = [x for x in self.params.keys() if x != 'best_dev_score']
        return self._param_names

    def update(self, args, dev_score):
        """ If dev_score is better than best on record, update the parameters and save to file. """
        if dev_score > self.params['best_dev_score']:
            for p in self.param_names:
                self.params[p] = args[p]
            self.params['best_dev_score'] = dev_score
            save_param(self.params, self.param_file)
            print("[Best parameters saved to file.]")
        return

    def load_to_args(self, args):
        """ Load optimal parameters into args (passed in); return new args. """
        for p in self.param_names:
            args[p] = self.params[p]
        return args
