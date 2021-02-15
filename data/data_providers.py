from .brats2020 import Brats2020DataProvider

class DataProviderHub:

    def __init__(self):
        self.ProviderHub = {
            'brats2020': Brats2020DataProvider,
        }

    def __getitem__(self, key):
        key = key.split('_')
        data_source, args = key[0], key[1:]
        args = '_'.join(args)
        return self.ProviderHub[data_source], args
