from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML



# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

__C = ConfigurationNode()
__C.CONFIG_PATH = './config/voc/dependency_graph_lr.yaml'
__C.OUTPUT_PATH = './outputs/'
__C.LOGGER_PATH = "./logger/"
__C.DEBUG = False
__C.NUM_POOLS = 10
__C.DEVICE = "gpu"
# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #

__C.DATASET = ConfigurationNode()
__C.DATASET.TRAIN_FILE_LOCATION = 'data/external/cnn_outputs/train_saved_data.0.txt'
__C.DATASET.TEST_FILE_LOCATION = 'data/external/cnn_outputs/val_saved_data.0.txt'

# __C.DATASET.AUGMENTATION.GAUSS_VAR_LIMIT =(10.0, 40.0)
# __C.DATASET.AUGMENTATION.BLUR_LIMIT = 7

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #

__C.TRAIN = ConfigurationNode()

# If True Train the model, else skip training.
__C.TRAIN.ENABLE = True

# Dataset.
__C.TRAIN.DATASET = "voc"

# Total mini-batch size.
__C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
__C.TRAIN.EVAL_PERIOD = 5

# Save model checkpoint every checkpoint period epochs.
__C.TRAIN.CHECKPOINT_PERIOD = 10

# Path to the checkpoint to load the initial weight.
__C.TRAIN.CHECKPOINT_FILE_PATH = "models"

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #

# Train model definitions
# model backbone wetlab
__C.MODEL = ConfigurationNode()
__C.MODEL.NAME = "nn"
__C.MODEL.EPOCHS = 200
__C.MODEL.LEARNING_RATE = 0.01
__C.MODEL.L1_WEIGHT = 0.01
__C.MODEL.SAVE_MODEL_PATH = "./models/"
__C.MODEL.DIRECTORY = ""

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #

__C.TEST = ConfigurationNode()

# If True test the model, else skip the testing.
__C.TEST.ENABLE = True

# Dataset for testing.
__C.TEST.DATASET = "voc"

# Total mini-batch size
__C.TEST.BATCH_SIZE = 1024

# Path to the checkpoint to load the initial weight.
__C.TEST.CHECKPOINT_FILE_PATH = ""

# Path to saving prediction results file.
__C.TEST.SAVE_RESULTS_PATH = ""

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
__C.SOLVER = ConfigurationNode()

# Base learning rate.
__C.SOLVER.BASE_LR = 0.1

# Final learning rates for 'cosine' policy.
__C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
__C.SOLVER.GAMMA = 0.1

# Maximal number of epochs.
__C.SOLVER.MAX_EPOCH = 300

# Momentum.
__C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
__C.SOLVER.DAMPENING = 0.0


__C.SOLVER.CLIP_GRAD_VAL = None
__C.SOLVER.CLIP_GRAD_L2NORM = 2.0


# Nesterov momentum.
__C.SOLVER.NESTEROV = False

# L2 regularization.
__C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
__C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
__C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
__C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
__C.SOLVER.OPTIMIZING_METHOD = "sgd"

__C.DEBUG = False

__C.JOINT_LEARNING = ConfigurationNode()

__C.JOINT_LEARNING.PRETRAINED = False
__C.JOINT_LEARNING.MODEL_DIRECTORY = ""
__C.JOINT_LEARNING.LEARNING_RATE = 0.01
__C.JOINT_LEARNING.L1_WEIGHT = 0.001
__C.JOINT_LEARNING.EXTRA_EPOCHS = 3
__C.JOINT_LEARNING.TWO_LOSSES = False
__C.JOINT_LEARNING.WEIGHT_DECAY = 0.001
__C.JOINT_LEARNING.DN_TYPE = ""

# __C.JOINT_LEARNING =
# Add custom config with default values.
# custom_config.add_custom_config(__C)

__C.DN_INFERENCE = ConfigurationNode()

__C.DN_INFERENCE.NUM_SAMPLES = 5000
__C.DN_INFERENCE.NUM_POOLS = 10
__C.DN_INFERENCE.BATCH_SIZE = 1024


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()


def assert_and_infer_cfg(cfg):
    # Some assert statements to see if correct arguments are given
    # # BN assertions.
    # if cfg.BN.USE_PRECISE_STATS:
    #     assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # # TRAIN assertions.
    # assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    # assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0
    #
    # # TEST assertions.
    # assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    # assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    #
    # # RESNET assertions.
    # assert cfg.RESNET.NUM_GROUPS > 0
    # assert cfg.RESNET.WIDTH_PER_GROUP > 0
    # assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0
    #
    # # Execute LR scaling by num_shards.
    # if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
    #     cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
    #     cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
    #     cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS
    #
    # # General assertions.
    # assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg
