import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4

# load the pure pytorch module:
torch_module = EEGNetv4(in_chans=3, n_classes=2, input_window_samples=200)
torch_module.load_state_dict(torch.load(local_paths['torch']))

# load the pure pytorch module:
skorch_module = EEGNetv4(in_chans=3, n_classes=2, input_window_samples=200)
skorch_classifier = EEGClassifier(skorch_module, max_epochs=5)
skorch_classifier.initialize()
skorch_classifier.load_params(
    f_params=local_paths['f_params'],
    f_optimizer=local_paths['f_optimizer'],
    f_history=local_paths['f_history'],)