"""
@author: Deepak Ravikumar Tatachar
@copyright: Nanoelectronics Research Laboratory
"""

import os
import multiprocessing

def main():
    import argparse
    import torch
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from utils.save import TrainingState
    from utils.instantiate_model import instantiate_model

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset',                default='CIFAR10',      type=str,       help='Set dataset to use')
    parser.add_argument('--parallel',               default=False,          type=str2bool,  help='Device in  parallel')
    parser.add_argument('--test_batch_size',        default=512,            type=int,       help='Test batch size')
    parser.add_argument('--arch',                   default='resnet18',     type=str,       help='Network architecture')
    parser.add_argument('--suffix',                 default='erm_1_std_tinyimagenet',
                                                                            type=str,       help='Appended to model name')


    global args
    args = parser.parse_args()
    print(args)

    # Setup right device to run on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Use the following transform for training and testing
    print('\n')
    dataset = load_dataset(dataset=args.dataset,
                           test_batch_size=args.test_batch_size,
                           device=device,
                           val_split=0.1)

    # Instantiate model 
    net, model_name = instantiate_model(dataset=dataset,
                                        arch=args.arch,
                                        suffix=args.suffix,
                                        device=device,
                                        path='./pretrained/',
                                        load=True)

    net.eval()
    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    print(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))

    train_correct, train_total, train_accuracy = inference(net=net, data_loader=dataset.train_loader, device=device)
    print(' Train set: Accuracy: {}/{} ({:.2f}%)'.format(train_correct, train_total, train_accuracy))

    val_correct, val_total, val_accuracy = inference(net=net, data_loader=dataset.val_loader, device=device)
    print(' Val set: Accuracy: {}/{} ({:.2f}%)'.format(val_correct, val_total, val_accuracy))

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()