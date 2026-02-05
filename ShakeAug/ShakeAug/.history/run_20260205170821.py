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
    if name in ["mnist", "cifar10"]:
        return 10
    if name == "cifar100":
        return 100
    if name == "imagenet":
        return 1000
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main Program')

    # basic config
    parser.add_argument('--dataset', type=str, default = 'cifar10',choices = ['mnist', 'cifar10', 'cifar100', 'imagenet'], help = 'dataset name')
    parser.add_argument('--model', type=str, default = 'ResNet_18', help='model name')
    parser.add_argument('--checkpoints', type=str, default = './checkpoints', help='checkpoints path')
    parser.add_argument('--aug_mode', type=str, default='traditional', choices=['none', 'traditional', 'shake'], help='augmentation mode: none (baseline), traditional (typical augmentation), shake (our method)')
    
    # data paths
    parser.add_argument('--train_data_path', type=str, default='./Train', help='train dataset path')
    parser.add_argument('--test_data_path', type=str, default='./Test', help='test dataset path')

    # image size and number classes
    parser.add_argument('--img_size', type=int, default = 224, help='image size for imagenet dataset, other dataset is fixed, when network is densenet, image is 256, other is 224')
    parser.add_argument('--num_class', type=int, default=None, help='number of classes; if None, infer from dataset')

    # optimization
    parser.add_argument('--train_epochs', type=int, default = 20, help='training epochs')
    parser.add_argument('--learning_rate', type=float, default = 1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default = 32, help='batch size')
    parser.add_argument('--lradj', type=str, default = 'cosine', help='adjust learning rate')
    parser.add_argument('--num_workers', type=int, default = 8, help='data loader num workers')

    # early stopping
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--earlystop_delta', type=float, default=0.0, help='early stopping delta')

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu if available')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id (single gpu)')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids, e.g. "0,1,2,3"')

    args = parser.parse_args()

    # respect user's --use_gpu flag; only disable if cuda unavailable
    if args.use_gpu and not torch.cuda.is_available():
        args.use_gpu = False
    print("CUDA available:", torch.cuda.is_available(), "| use_gpu:", args.use_gpu)

    # multi-gpu parsing
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(i) for i in device_ids]
        args.gpu = args.device_ids[0]
    
    # infer num_class if not set
    if args.num_class is None:
        inferred = infer_num_class(args.dataset)
        if inferred is None:
            raise ValueError(f"Cannot infer num_class for dataset={args.dataset}. Please set --num_class.")
        args.num_class = inferred
    
    print("Args in experiment:")
    print(args)
    
    setting = 'dataset_{}_model_{}_lr{}_bs{}_epochs{}'.format(
        args.dataset,
        args.model,
        args.learning_rate,
        args.batch_size,
        args.train_epochs
    )

    print('>>>>>>>>>>>>>>>>Start Training>>>>>>>>>>>>>>>')
    print(f'Setting: {setting}')
    if args.use_multi_gpu and args.use_gpu:
        print(f'Devices: {args.devices}')
    else:
        print(f'GPU: {args.gpu}')
    print(f'Training epochs: {args.train_epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Image size: {args.img_size}')
    print(f'num_class: {args.num_class}')

    exp = Experiment_Main(args)
    exp.train(setting)

    print('>>>>>>>>>>>>>>>>Finishing>>>>>>>>>>>>>>>')






