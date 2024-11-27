import yaml

class LTTMLConfig:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None

        self.model_type = config.get('model')
        self.num_outputs = config.get('num_outputs')
        self.batch_size = config.get('batch_size') # int(sys.argv[1]) if len(sys.argv) > 1 else 1
        self.epochs = config.get('epochs') # int(sys.argv[2]) if len(sys.argv) > 2 else 1
        self.use_mps = config.get('use_mps')
        self.use_cuda = config.get('use_cuda')
        self.data_dir = config.get('data_directory')
        self.learning_rate = config.get('learning_rate')
        self.augmentation = config.get('augmentation') #8 # Amount of copies to produce with different time dimension shifts
        self.mlflow_user = config.get("mlflow_user")
        self.mlflow_pw = config.get("mlflow_pw")
        self.mlflow_uri = config.get("mlflow_uri")
        self.mlflow_expname = config.get("mlflow_expname")
        self.mlflow_description = config.get("mlflow_description")

        self.normalize = config.get('normalize')
        self.calculate_norm_stats = config.get('calculate_norm_stats')

        self.min_examples_per_class = config.get('min_examples_per_class')
        self.max_examples_per_class = config.get('max_examples_per_class')
        self.conv1_size = config.get('conv1_size')
        self.conv2_size = config.get('conv2_size')
        self.hidden = config.get('hidden')
        self.dropout_rate = config.get('dropout_rate')
        self.spectrogram_samples = config.get('spectrogram_samples')
        self.spectrogram_bins = config.get('spectrogram_bins')

        self.mlflow_runname_prefix = config.get('mlflow_runname_prefix')

        self.best_model = config.get('best_model')
        self.use_softmax = config.get('use_softmax')

        self.max_classes = config.get('max_classes')

        self.normalization_mean = config.get('normalization_mean')
        self.normalization_std = config.get('normalization_std')

        self.use_mlflow = config.get('use_mlflow')
