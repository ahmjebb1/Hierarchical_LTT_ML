from cnn import BirdCNNModel
from transformers import AutoModelForAudioClassification

def get_model(config):
    if config.model_type == 'cnn':
        return BirdCNNModel(config)
    elif config.model_type == 'transformer':
        return AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593") #TransformerModel()
    # elif model_type == 'svm':
        # return SVMModel()
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")