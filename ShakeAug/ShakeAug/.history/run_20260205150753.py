import argparse
import torch
import warnings
from exp.exp_main import Experiment_Main
warnings.filterwarnings('ignore')

def str2bool(v):
    """Fix argparse bool parsing: allow true/false/1/0/yes/no."""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def infer_num_class(dataset_name: str):
    name = dataset_name.lower()
    if name in ["mnist", "minist", "cifar10"]:
        return 10
    if name == "cifar100":
        return 100
    if name == "imagenet":
        return 1000
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main Program')

    # basic config
    parser.add_argument('--dataset', type=str, default = 'cifar10',choices = ['mnist', 'minist', 'cifar10', 'cifar100', 'imagenet'], help = 'dataset name')
    parser.add_argument('--model', type=str, default = 'ResNet_18', help='model name')
    parser.add_argument('--checkpoints', type=str, default = './checkpoints', help='checkpoints path')

    # data paths
    parser.add_argument('--train_data_path', type=str, default='./Train', help='train dataset path')
    parser.add_argument('--test_data_path', type=str, default='./Test', help='test dataset path')

    # image size and number classes
    parser.add_argument('--img_size', type=int, default = 224, help='image size (single int). Example: 224')
    parser.add_argument('--num_class', type=int, default=None, help='number of classes; if None, infer from dataset')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--target_acc', type=float, default=None, help='target accuracy (optional)')
    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')

    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')

    # early stopping
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--earlystop_delta', type=float, default=0.0, help='early stopping delta')

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu if available')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus, e.g. "0,1,2,3"')


    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() else False
    print("CUDA available:", torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if isinstance(args.img_size, list):
        args.img_size = tuple(args.img_size)

    if isinstance(args.patch_size, list):
        args.patch_size = tuple(args.patch_size)
    
    print("Args in experiment:")
    print(args)
    
    setting = 'model_{}_lr{}_bs{}_epochs{}'.format(
        args.model,
        args.learning_rate,
        args.batch_size,
        args.train_epochs
    )

    print('>>>>>>>>>>>>>>>>Start Training>>>>>>>>>>>>>>>')
    print(f'Setting: {setting}')
    print(f'Device: {args.devices}')
    print(f'Training epochs: {args.train_epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Image size: {args.img_size}')
    print(f'Patch size: {args.patch_size}')


    exp = Experiment_Main(args)
    exp.train(setting)

    print('>>>>>>>>>>>>>>>>Finisging>>>>>>>>>>>>>>>')






