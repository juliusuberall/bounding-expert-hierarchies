# Class to register all relevant experiment data
class CoreRegistry:
    def __init__(self):
        self.registry = {}

    def get(self, name : str):
        if name in self.registry.keys():
            return self.registry[name]
        else:
            raise ValueError("metric {} not registered".format(name))

    def add(self, name : str, value):
        self.registry[name] = value

# Data directory managment
core_registration_keys = {
    'train_val_loss_key' : '_training_validation_loss', # Stores training validation loss, first value is epoch logging frequence
}