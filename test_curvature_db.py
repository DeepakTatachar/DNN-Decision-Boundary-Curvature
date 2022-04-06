
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd

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

# generate class1 as points inside a circle of radius 1
def generate_class1(size):
    points = np.random.uniform(-1.1, 1.1, size)
    remove_idxs = []
    for idx, pt in enumerate(points):
        if np.linalg.norm(pt) > 1:
            remove_idxs.append(idx)
            
    
    points = np.delete(points, remove_idxs, axis=0)
    return points

# generate class2 as points outside a circle of radius 1
def generate_class2(size):
    points = np.random.uniform(-1.4, 1.4, size)
    remove_idxs = []
    for idx, pt in enumerate(points):
        if np.linalg.norm(pt) < 1.3:
            remove_idxs.append(idx)
            
    
    points = np.delete(points, remove_idxs, axis=0)
    return points


set_size = 50

# train neural network to classify between c1 and c2
c1 = generate_class1((set_size * 3 , 2))
c2 = generate_class2((set_size * 3, 2))

data = np.concatenate((c1, c2), axis=0)
labels = np.concatenate((np.zeros(10), np.ones(10)))

data = np.concatenate((c1, c2), axis=0)
labels = np.concatenate((np.zeros(c1.shape[0]), np.ones(c2.shape[0])))

# define a 3 layer neural network
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.fc3 = torch.nn.Linear(50, 2)
        self.activation = torch.nn.ReLU()

    def forward(self, x,activation=None):
        act = self.activation if activation is None else activation
        x = self.fc1(x)
        x = act(x)
        x = self.fc2(x)
        x = act(x)
        x = self.fc3(x)
        return x

# loop over the data to train the neural network
net = MLP()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = torch.from_numpy(data).to(torch.float32).to(device)
labels = torch.from_numpy(labels).to(torch.long).to(device)
net.to(device)

for epoch in range(100):
    # forward pass
    outputs = net(data)
    loss = criterion(outputs, labels)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss.item()))

net.eval()
# plot the decision boundary

def make_meshgrid(x, y, h=.002):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, net, xx, yy, **params):
    data = np.c_[xx.ravel(), yy.ravel()]
    data = torch.Tensor(data).to(torch.float32).to(device)
    with torch.no_grad():
        out = net(data)
    pred = out.max(1)[1].reshape(xx.shape).cpu().numpy()
    
    Z = pred
    out1 = ax.contour(xx, yy, Z, **params, cmap=plt.cm.binary)
    out2 = ax.contourf(xx, yy, Z, **params, cmap=plt.cm.coolwarm)
    return out1, out2

def plot_db(data, labels, net):
    X0, X1 = data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy()
    xx, yy = make_meshgrid(X0, X1)

    fig, ax = plt.subplots()
    out1, out2 = plot_contours(ax, net, xx, yy, alpha=0.6)

    # act = torch.nn.Softplus()
    with torch.no_grad():
            out = net(data)

    conf = out.max(1)[0].cpu().numpy()

    sizes = conf *30
    scatter = ax.scatter(X0, X1, c=labels.cpu().numpy(), cmap=plt.cm.coolwarm, s=sizes, edgecolors='k')
    # ax.set_ylabel('y label here')
    # ax.set_xlabel('x label here')
    ax.set_xticks(())
    ax.set_yticks(())
    # ax.set_title(title)
    ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Class")

    return out1, out2, scatter
    

# plot_db(data, labels, net)

# We should see a decision boundary that is a circle
# plt.show()


# Now we will calculate the curvature of the decision boundary for each point
# We will use the gradient of the decision boundary to calculate the curvature


data_clone = data.clone()

target_labels = 1 - labels

data.requires_grad = True
optimizer = torch.optim.SGD([data], lr=1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

steps = 100

data_for_animation = []
     
for i in range(steps):
    optimizer.zero_grad()
    data.requires_grad = True
    # We are interated in the curvautre of the decision boundary with respect to the input space
    outputs = net(data)
    pred_labels = outputs.max(1)[1]
    loss =  criterion(outputs, pred_labels) + outputs.max(1)[0].mean() #outputs.max(1)[0] - outputs[:,target_labels]

    loss.backward()

    # grad gives the input direction perpendicular to the decision boundary
    df = data.grad.clone()
    df = -torch.nn.functional.normalize(df, p=2, dim=1)

    optimizer.step()
    # scheduler.step()

    if i % 10 == 0:
        data_for_animation.append((data.clone().detach(), df.clone().detach()))
    
idx = 0
while idx < len(data_for_animation):
    plot_data, plot_df = data_for_animation[idx]
    # plot df as an arrow in the direction of the gradient
    plot_db(plot_data, labels, net)
    plt.quiver(plot_data[:, 0].cpu().numpy(), plot_data[:, 1].cpu().numpy(), plot_df[:, 0].cpu().numpy(), plot_df[:, 1].cpu().numpy(), scale=80, width=0.01)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.savefig('./outputs/images/frame{:04d}.png'.format(idx))
    plt.close()

    idx += 1



'''
Function to calculate the hessian of the loss wrt to the input data
'''
def calc_hessian(data, loss):
    if data.shape[0] != 1:
        raise ValueError("data must be a single image")

    num_params = data.shape[1]
    hessian = torch.zeros(num_params, num_params).to(device)
    first_derivative, = torch.autograd.grad(loss, [data], create_graph=True)
    first_derivative = first_derivative.flatten()
    data.grad = torch.zeros_like(data)

    for param_idx in range(num_params):
        jacobian_vec = torch.zeros(num_params).to(device)
        jacobian_vec[param_idx] = 1.
        # clear data.grad
 
        data.grad.zero_()
        
        first_derivative.backward(jacobian_vec, retain_graph=True)
        hessian[:, param_idx] = data.grad.flatten()
    return first_derivative.detach(), hessian.detach()

'''
Function retuns the orthogonal vector to the input vector
'''
def compute_ortho(inp_vec):
    return

def maximize_curvature(hess, grad_mag, normal):
    v = torch.randn(1, hess.shape[0]).to(device)
    hess.requires_grad = False
    normal.requires_grad = False
    grad_mag.requires_grad = False
    
    optimizer = torch.optim.SGD([v], lr=0.8)
    normal = normal / grad_mag
    v = v - (v  @ normal) * normal.T
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    steps = 50
    for i in range(steps):
        
        v.requires_grad = True
        optimizer.zero_grad()
       
        cur = (v @ hess @ v.T) / (grad_mag * (v.norm() ** 2))
        
        loss = -cur
        
        loss.backward()
        optimizer.step()
        # scheduler.step()

    v = torch.nn.functional.normalize(v, p=2, dim=1)
    print(v @ normal)
    return v.detach(), cur.detach()
            

z = data
z.requires_grad = True
h = []
vs = []
r_crv = []
normals = []
hess_act = torch.nn.Softplus()
for idx in range(z.shape[0]):
    pt = z[idx].unsqueeze(0).clone().detach()
    pt.requires_grad = True
    out = net(pt, hess_act)
    pred_label = out.max(1)[1]

    top2 = torch.topk(out, 2, sorted=False)[0]
    loss =  (top2[:,0] - top2[:,1]).sum()#criterion(out, pred_label) #
    f_grad, hess = calc_hessian(pt, loss)

    grad = f_grad.unsqueeze(1)
    grad_mag = torch.norm(grad, p=2)

    # get v from the optimization
    # p = 1 - grad @ grad.T
    # u, s, v = torch.svd( p @ hess.T @ p / grad_mag)

    # r = (v @ hess @ v.T) / grad_mag

    normals.append(torch.nn.functional.normalize(grad.detach(), p=2, dim=0))
    v, r = maximize_curvature(hess.detach(), grad_mag.detach(), grad.detach())

    r_crv.append(r.squeeze())
    vs.append(v.squeeze())

r = torch.stack(r_crv)
v = torch.stack(vs)

print(r)

def plot_circles_of_curvature(z, r, v, normals):
    z.requires_grad = False
    # plot the circle for point z[0]
    plot_db(z, labels, net)
    plt.quiver(z[:, 0].cpu().numpy(), z[:, 1].cpu().numpy(), v[:, 0].cpu().numpy(), v[:, 1].cpu().numpy(), scale=80, width=0.01)

    circ = None
    for circle_idx in range(z.shape[0]):
        # the raduis of curvature of circle
        r_0 = r[circle_idx].cpu().numpy().item()

        # center of the circle is r from v
        c = r_0 * normals[circle_idx]

        x, y = z[circle_idx, 0] - c[0], z[circle_idx, 1] - c[1]
        x, y = x.item(), y.item()

        if circ:
            circ.set_radius(r_0)
            circ.set_center((x, y))
        else:
            circ = plt.Circle((x, y), r_0, color='r', fill=False, linestyle='-')
            plt.gca().add_artist(circ)
            

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

plot_circles_of_curvature(z, r, v, normals)
plt.hist(r.cpu().numpy(), bins=35)   
plt.show()


temp = 55