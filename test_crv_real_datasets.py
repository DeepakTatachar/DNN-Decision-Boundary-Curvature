
import os
import multiprocessing

def main():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.str2bool import str2bool
    from utils.load_dataset import load_dataset
    from utils.instantiate_model import instantiate_model
    import argparse

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset',                default='CIFAR10',      type=str,       help='Set dataset to use')
    parser.add_argument('--parallel',               default=False,          type=str2bool,  help='Device in  parallel')
    parser.add_argument('--test_batch_size',        default=512,            type=int,       help='Test batch size')
    parser.add_argument('--arch',                   default='resnet18',     type=str,       help='Network architecture')
    parser.add_argument('--suffix',                 default='1',
                                                                            type=str,       help='Appended to model name')

    global args
    args = parser.parse_args()
    print(args)


    seed_val = 35
    # Set numpy random seed
    np.random.seed(seed_val)

    # Set torch random seed
    torch.manual_seed(seed_val)

    # set cuda backend seeds
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loop over the data to train the neural network
    dataset = load_dataset(dataset=args.dataset,
                           train_batch_size=128,
                           test_batch_size=args.test_batch_size,
                           device=device,
                           val_split=0.1,
                           data_percentage=0.5,
                           random_seed=0)


    # Instantiate model 
    net_relu, model_name = instantiate_model(dataset=dataset,
                                             arch=args.arch,
                                             suffix=args.suffix,
                                             device=device,
                                             path='./pretrained/',
                                             load=True)

    optimizer = torch.optim.SGD(net_relu.parameters(), lr=0.5)

    net_relu.eval()
    steps = 100


    # Now we will calculate the curvature of the decision boundary for each point
    # We will use the gradient of the decision boundary to calculate the curvature
    try:
        all_data = torch.load("./outputs/{}_z.data".format(model_name))
    except:
        all_data = None
        for data, _ in dataset.train_loader:
            data = data.to(device)
            data.requires_grad = True
            optimizer = torch.optim.SGD([data], lr=1)

                
            for i in range(steps):
                optimizer.zero_grad()
                data.requires_grad = True
                # We are interated in the curvautre of the decision boundary with respect to the input space
                outputs = net_relu(data)
                # pred_labels = outputs.max(1)[1]
                top2 = torch.topk(outputs, 2, sorted=False)[0]
                loss =  (top2[:,0] - top2[:,1]).sum()
                # loss = criterion(outputs, pred_labels) + outputs.max(1)[0].mean() #outputs.max(1)[0] - outputs[:,target_labels]

                loss.backward()

                optimizer.step()

            if all_data == None:
                all_data = data.clone()
            else:
                all_data = torch.cat((all_data, data.clone()), 0)

            torch.save(all_data, "./outputs/{}_z.data".format(model_name))
    
    print(all_data.shape)
    '''
    Function to calculate the hessian of the loss wrt to the input data
    '''
    def calc_hessian(data, loss):


        num_params = data.view(data.shape[0], -1).shape[1]
        hessian = torch.zeros(data.shape[0], num_params, num_params).to(device)
        first_derivative, = torch.autograd.grad(loss, [data], create_graph=True)
        first_derivative = first_derivative.view(data.shape[0], -1)
        data.grad = torch.zeros_like(data)

        for param_idx in range(num_params):
            jacobian_vec = torch.zeros(data.shape[0], num_params).to(device)
            jacobian_vec[:, param_idx] = 1.
            # clear data.grad
    
            data.grad.zero_()
            
            first_derivative.backward(jacobian_vec, retain_graph=True)
            hessian[:, :, param_idx] = data.grad.view(data.shape[0], -1)
        return first_derivative.detach(), hessian.detach()


    def maximize_curvature(hess, normal):
        v = torch.randn(normal.shape[0],  hess.shape[1]).to(device)
        hess.requires_grad = False
        normal.requires_grad = False
        
        optimizer = torch.optim.SGD([v], lr=0.8)
        grad_mag = torch.norm(normal, p=2, dim=1)
        
        normal = torch.nn.functional.normalize(normal, p=2, dim=1)

        normal_dot = (v * normal).sum(dim=1)
        normal_dot = torch.stack([normal_dot] * normal.shape[1], dim=1)
        v = v - normal_dot * normal
        v = v.unsqueeze(1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        steps = 50
        for i in range(steps):
            
            v.requires_grad = True
            optimizer.zero_grad()
        
            num = v @ hess @ torch.transpose(v, 2, 1)
            den = grad_mag * (torch.norm(v, p=2, dim=(1,2)) ** 2)
            cur = num.squeeze() / den
            
            loss = -(cur.sum())
            
            loss.backward()
            optimizer.step()
            # scheduler.step()

        v = torch.nn.functional.normalize(v, p=2, dim=(1,2)).squeeze(1)
        print((v * normal).sum(dim=1))
        return v.detach(), cur.detach()
                

    z = all_data
    z.requires_grad = True
    vs = []
    r_crv = []
    normals = []
    hess_act = torch.nn.Softplus()
    model_args = {'act': hess_act}
    net_hess, model_name = instantiate_model(dataset=dataset,
                                             arch=args.arch,
                                             suffix=args.suffix,
                                             device=device,
                                             path='./pretrained/',
                                             load=True,
                                             model_args=model_args)
    
    net_hess.to(device)
    net_hess.eval()
    idx = 0
    batch_size = dataset.train_batch_size
    while idx < z.shape[0]:
        pt = z[idx:idx+batch_size].clone().detach()
        pt.requires_grad = True
        out = net_hess(pt)


        top2 = torch.topk(out, 2, sorted=False)[0]
        loss =  (top2[:,0] - top2[:,1]).sum()
        f_grad, hess = calc_hessian(pt, loss)

        normals.append(torch.nn.functional.normalize(f_grad.detach(), p=2, dim=0))
        v, r = maximize_curvature(hess.detach(), f_grad.detach())

        r_crv.append(r.squeeze())
        vs.append(v.squeeze())

        idx += batch_size

    r = torch.stack(r_crv)
    v = torch.stack(vs)

    print(r)
    torch.save(r, './outputs/{}_r.data'.format(model_name))
    plt.hist(r.cpu().numpy(), bins=35)   
    plt.savefig('./outputs/{}_r.png'.format(model_name))
    plt.show()

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()