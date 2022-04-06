  
import torch
import numpy as np
  
class TrainingState:
    def __init__(self, dataset_name, model_name, parallel=False):
        self.state_save_path = './pretrained/'+ dataset_name.lower() + '/temp/' + model_name  + '.temp'
        self.model_save_path = './pretrained/'+ dataset_name.lower() + '/' + model_name  + '.ckpt'
        self.raw_save_path = './pretrained/'+ dataset_name.lower() + '/'  + '{}.raw'
    
    '''
    Function to save training state and model
    '''
    def save_training_state(self, 
                            net, 
                            optimizer,
                            epoch,
                            best_val_acc,
                            best_val_loss,
                            scheduler,
                            kwargs=None,
                            parallel=False):
        
        saved_training_state = {    'epoch'     : epoch + 1,
                                    'optimizer' : optimizer.state_dict(),                                       
                                    'scheduler' : scheduler.state_dict() if scheduler is not None else None,
                                    'best_val_accuracy' : best_val_acc,
                                    'best_val_loss' : best_val_loss,
                                    'kwargs' : kwargs
                                }
        if parallel:
            saved_training_state['model'] = net.module.state_dict(),
            
        else:
            saved_training_state['model'] = net.state_dict()

        torch.save(saved_training_state, self.state_save_path)
    
    '''
    Function to save model state
    '''
    def save_model(self, net, parallel=False):
        if parallel:
            torch.save(net.module.state_dict(), self.model_save_path)
        else:
            torch.save(net.state_dict(), self.model_save_path)

    '''
    Function to load training state
    '''        
    
    def load_training_state(self, net, optimizer=None, scheduler=None, parallel=False, override=False):
        try:
            saved_training_state = torch.load(self.state_save_path)
            print("Saved state found")

            if override:
                epoch = 0
                best_val_acc = 0
                best_val_loss = np.Inf
            else:
                print("Loading saved state")
                if parallel:
                    net.module.load_state_dict(saved_training_state['model'])

                else:
                    net.load_state_dict(saved_training_state['model'])

                if optimizer:
                    optimizer.load_state_dict(saved_training_state['optimizer'])

                if scheduler:
                    scheduler.load_state_dict(saved_training_state['scheduler'])

                epoch = saved_training_state['epoch']
                best_val_acc = saved_training_state['best_val_accuracy']
                best_val_loss = saved_training_state['best_val_loss']
                kwargs = saved_training_state['kwargs']
        except:
            # No saved state found
            print("Warning: No saved state found")
            print("Creating new save state")
            epoch = 0
            best_val_acc = 0
            best_val_loss = 0
            kwargs = None

        return epoch, best_val_acc, best_val_loss, kwargs

    def save_raw_state(self, state, name):
        torch.save(state, self.raw_save_path.format(name))

    def load_raw_state(self, name):
        try:
            return torch.load(self.raw_save_path.format(name))
        except:
            return None

        