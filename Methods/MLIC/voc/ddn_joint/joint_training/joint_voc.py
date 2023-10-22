import argparse

import wandb

from engine import *
from models import *
from DataLoader.voc import *

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-graph_file', default='./data/voc/voc_adj.pkl', type=str, metavar='PATH',
                    help='path to graph (default: none)')
parser.add_argument('-word_file', default='./data/voc/voc_glove_word2vec.pkl', type=str, metavar='PATH',
                    help='path to word (default: none)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='checkpoint/voc_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', type=int, default=1,
                    help='use pre-trained model')
parser.add_argument('-pm', '--pretrain_model', default='./pretrained/resnet101.pth.tar', type=str, metavar='PATH',
                    help='path to latest pretrained_model (default: none)')
parser.add_argument('--pool_ratio', '-po', default=0.2, type=float, metavar='O',
                    help='ratio of node pooling (default: 0.2)')
parser.add_argument('--backbone', '-bb', default='resnet101', type=str, metavar='B',
                    help='backbone of the model')
# Dataset to be used
parser.add_argument('--val', action='store_true', default=False,
                    help='Get outputs for train set or validation set')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Check if we are in debug mode')

# DDN
parser.add_argument('--dn_type', help='Type of DDN you want to use', default='nn', choices=['nn', 'lr'])

@logger.catch
def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    if args.debug:
        wandb.init(project=f"DEBUG Joint model VOC {args.dn_type}")
        print("We are in debug mode")
    else:
        wandb.init(project=f"Joint model VOC {args.dn_type}")
    wandb.config.update(args)

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = Voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
    val_dataset = Voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')
    num_classes = 20
    # load model
    # model = SGDNN(num_classes=num_classes, t=0.4, adj_file='data/voc/voc_adj.pkl')
    model = MSGDN(num_classes, args.pool_ratio, args.backbone, args.graph_file)

    if args.pretrained:
        model = load_pretrain_model(model, args)
    model.cuda()
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/voc2007/{date}/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['train_split'] = True
    state['dataset'] = 'voc'
    state['args'] = args
    state['start_epoch'] = args.start_epoch

    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    get_op_on_train_set = not args.val
    if get_op_on_train_set:
        # We only need this part to get outputs for training the pipeline model
        state['evaluate'] = True
        engine.learning(model, criterion, train_dataset, train_dataset, optimizer)
    else:
        engine.state['train_split'] = False
        engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main_voc2007()