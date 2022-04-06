"""
@author: Deepak Ravikumar Tatachar
@copyright: Nanoelectronics Research Laboratory
"""

import os
import multiprocessing

def main():
    import argparse
    import torch
    import torch.nn as nn
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from utils.averagemeter import AverageMeter
    from utils.instantiate_model import instantiate_model
    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--epochs',                 default=250,            type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='CIFAR10',      type=str,       help='Set dataset to use')
    parser.add_argument('--parallel',               default=False,          type=str2bool,  help='Device in  parallel')
    parser.add_argument('--lr',                     default=0.01,           type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--optimizer',              default='SGD',          type=str,       help='Optimizer for training')
    parser.add_argument('--loss',                   default='crossentropy', type=str,       help='Loss function for training')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=2048,           type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=512,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.1,            type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=True,           type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=True,           type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initialising the seed for reproducibility')

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialisation')
    parser.add_argument('--suffix',                 default='1',            type=str,       help='Appended to model name')
    parser.add_argument('--arch',                   default='resnet18',     type=str,       help='Network architecture')

    # Summary Writer Tensorboard
    parser.add_argument('--comment',                default="",             type=str,       help='Comment for tensorboard')

    global args
    args = parser.parse_args()
    print(args)

    # Parameters
    num_epochs = args.epochs
    learning_rate = args.lr

    # Setup right device to run on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Use the following transform for training and testing
    print('\n')
    dataset = load_dataset(dataset=args.dataset,
                           train_batch_size=args.train_batch_size,
                           test_batch_size=args.test_batch_size,
                           val_split=args.val_split,
                           augment=args.augment,
                           padding_crop=args.padding_crop,
                           shuffle=args.shuffle,
                           random_seed=args.random_seed,
                           device=device,
                           data_percentage=0.5)

    # Instantiate model 
    net, model_name = instantiate_model(dataset=dataset,
                                        arch=args.arch,
                                        suffix=args.suffix,
                                        load=args.resume,
                                        torch_weights=False,
                                        device=device,
                                        path='./pretrained/')

    if args.use_seed:  
        if args.save_seed:
            print("Saving Seed")
            torch.save(net.state_dict(),'./seed/' + args.dataset.lower() + '_' + args.arch + ".Seed")
        else:
            print("Loading Seed")
            net.load_state_dict(torch.load('./seed/'+ args.dataset.lower() +'_' + args.arch + ".Seed"))
    else:
        print("Random Initialisation")

    # Optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    weight_decay=5e-4)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(),
                                        lr=learning_rate)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=learning_rate)
    else:
        raise ValueError ("Unsupported Optimizer")

    # Loss
    if args.loss.lower() == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss.lower() == 'mse':
        criterion=torch.nn.MSELoss()
    else:
        raise ValueError ("Unsupported loss function")

    if args.resume:
        saved_training_state = torch.load('./pretrained/'+ args.dataset.lower()+'/temp/' + model_name  + '.temp')
        start_epoch =  saved_training_state['epoch']
        optimizer.load_state_dict(saved_training_state['optimizer'])
        net.load_state_dict( saved_training_state['model'])
        best_val_accuracy = saved_training_state['best_val_accuracy']
        best_val_loss = saved_training_state['best_val_loss']
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
    if args.parallel:
        net = nn.DataParallel(net, device_ids=[0,1])
    else:
        net = net.to(device)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],
                                                     gamma=0.1)

    writer = SummaryWriter(comment=args.comment)

    # Train model
    for epoch in range(start_epoch, num_epochs, 1):
        net.train()
        train_correct = 0.0
        train_total = 0.0
        save_ckpt = False
        losses = AverageMeter('Loss', ':.4e')
        print('')
        for batch_idx, (data, labels) in enumerate(dataset.train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            # Clears gradients of all the parameter tensors
            optimizer.zero_grad()
            out = net(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            train_correct += (out.max(-1)[1] == labels).sum().long().item()
            train_total += labels.shape[0]

            if batch_idx % 48 == 0:
                trainset_len = (1 - args.val_split) * len(dataset.train_loader.dataset)
                curr_acc = 100. * train_total / trainset_len
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               train_total,
                                                                               trainset_len,
                                                                               curr_acc,
                                                                               losses.avg))

        train_accuracy = float(train_correct) * 100.0 / float(train_total)
        print('Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                                  train_correct,
                                                                                  train_total,
                                                                                  train_accuracy,
                                                                                  losses.avg))

        writer.add_scalar('Loss/train', losses.avg, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        
        # Step the scheduler by 1 after each epoch
        scheduler.step()
        
        if args.val_split > 0.0: 
            val_correct, val_total, val_accuracy, val_loss = inference(net=net,
                                                                       data_loader=dataset.val_loader,
                                                                       device=device,
                                                                       loss=criterion)

            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy 
                best_val_loss = best_val_loss
                save_ckpt = True
        else: 
            val_accuracy= float('inf')
            if (epoch + 1) % 10 == 0:
                save_ckpt = True

        if args.parallel:
            saved_training_state = {    'epoch'     : epoch + 1,
                                        'optimizer' : optimizer.state_dict(),
                                        'model'     : net.module.state_dict(),
                                        'best_val_accuracy' : best_val_accuracy,
                                        'best_val_loss' : best_val_loss
                                    }
        else:
            saved_training_state = {    'epoch'     : epoch + 1,
                                        'optimizer' : optimizer.state_dict(),
                                        'model'     : net.state_dict(),
                                        'best_val_accuracy' : best_val_accuracy,
                                        'best_val_loss' : best_val_loss
                                    }

        torch.save(saved_training_state, './pretrained/'+ args.dataset.lower() + '/temp/' + model_name  + '.temp')
        
        if save_ckpt:
            print("Saving checkpoint...")
            if args.parallel:
                torch.save(net.module.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name  + '.ckpt')
            else:
                torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')
            if args.test_accuracy_display:
                # Test model
                # Set the model to eval mode
                test_correct, test_total, test_accuracy = inference(net=net,
                                                                    data_loader=dataset.test_loader,
                                                                    device=device)

                print(" Training set accuracy: {}/{}({:.2f}%) \n" 
                      " Validation set accuracy: {}/{}({:.2f}%)\n"
                      " Test set: Accuracy: {}/{} ({:.2f}%)".format(train_correct,
                                                                    train_total,
                                                                    train_accuracy,
                                                                    val_correct,
                                                                    val_total,
                                                                    val_accuracy,
                                                                    test_correct,
                                                                    test_total,
                                                                    test_accuracy))

    # Test model
    # Set the model to eval mode
    print("\nEnd of training without reusing Validation set")
    if args.val_split > 0.0:
        print('Loading the best model on validation set')
        net.load_state_dict(torch.load('./pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt'))
        net = net.to(device)
        val_correct, val_total, val_accuracy = inference(net=net, data_loader=dataset.val_loader, device=device)
        print(' Validation set: Accuracy: {}/{} ({:.2f}%)'.format(val_correct, val_total, val_accuracy))
    else:
        print('Saving the final model')
        if args.parallel:
            torch.save(net.module.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name  + '.ckpt')
        else:
            torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')

    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    print(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))

    train_correct, train_total, train_accuracy = inference(net=net, data_loader=dataset.train_loader, device=device)
    print(' Train set: Accuracy: {}/{} ({:.2f}%)'.format(train_correct, train_total, train_accuracy))

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()