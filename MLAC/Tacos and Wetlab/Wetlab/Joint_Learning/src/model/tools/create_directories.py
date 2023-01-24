import os


def create_required_directories(args):
    # Make output dir and its parents if they do not exist
    if not os.path.exists(args.OUTPUT_PATH):
        os.mkdir(args.OUTPUT_PATH)

    if not os.path.exists(args.LOGGER_PATH):
        os.mkdir(args.LOGGER_PATH)

    # Make backup folders if they do not exist
    backup_dir = os.path.join(args.OUTPUT_PATH, 'model_backups')
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)

    # Make result folders if they do not exist
    results_dir = os.path.join(args.OUTPUT_PATH, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

