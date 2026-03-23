# CAGrad-Fast implementation as a multi-task learning optimizer wrapper from Conflict-Averse Gradient Descent for 
# Multi-task Learning by Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, Qiang Liu in 2021. https://doi.org/10.48550/arXiv.2110.14048

import torch
from torch.optim.optimizer import Optimizer

class CAGradOptimizer(Optimizer):
    def __init__(self, params, losses, lr=1e-3, c=0.5): # Experiment with c = {0.1, 0.5, 0.9}
        defaults = dict(lr=lr, c=c)
        super().__init__(params, defaults)
        self.num_tasks = None

    def _get_flat_grads_from_param_groups(self):
        """
        Flattens and concatenates gradients from all parameters into a single vector.
        """
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    # Create a zero tensor with same shape and device
                    view = p.data.new(p.data.numel()).zero_()
                else:
                    view = p.grad.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)
    
    def _solve_for_w_gpu(self, task_grads, avg_grad, phi):
        """
        Solves for optimal w approximately using a few steps of gradient descent.
        """
        K = self.num_tasks
        
        # Precalculate Gram matrix (g_i^T g_j) and g_i^T g_0
        GG = task_grads @ task_grads.T
        Gg = (task_grads @ avg_grad).unsqueeze(1)
        
        w_unconstrained = torch.zeros(K, 1, device=task_grads.device, requires_grad=True)
        w_optimizer = torch.optim.SGD([w_unconstrained], lr=25, momentum=0.5) # High lr to find minimum w faster

        # term c||g0|| from paper is simply sqrt(phi)
        c_prime = torch.sqrt(phi) 
        
        # Perform optimization steps
        num_w_steps = 20
        for _ in range(num_w_steps):
            w_optimizer.zero_grad()
            
            # Apply softmax to get weights w on probability simplex
            w = torch.softmax(w_unconstrained, dim=0)
            
            # Calculate objective function F(w) using precalculated matrices
            gw_sq_norm = w.T @ GG @ w
            gw_g0 = w.T @ Gg
            
            objective = gw_g0 + c_prime * torch.sqrt(gw_sq_norm + 1e-8)
            
            objective.backward()
            w_optimizer.step()

        # Return final weights
        return torch.softmax(w_unconstrained, dim=0).detach().view(-1)

    def step(self, task_losses):

        losses = [task_losses[i] for i in sorted(task_losses.keys())]

        self.num_tasks = len(losses)
        K = self.num_tasks

        # Calculate loss gradients for each task
        task_grads = []
        for i in range(K):
            self.zero_grad() 
            losses[i].backward(retain_graph=(i < K - 1)) # retain_graph for all but last
            flat_grad = self._get_flat_grads_from_param_groups()
            task_grads.append(flat_grad)

        task_grads = torch.stack(task_grads) # [K, num_params]

        # Compute g_0 and phi
        avg_grad = torch.mean(task_grads, dim=0)
        phi = self.param_groups[0]['c']**2 * torch.dot(avg_grad, avg_grad) # phi = c^2 * ||g_0||_2^2
        # NOTE: For 1D tensor, L2 norm = dot prod to tensor. 

        # Compute optimal weights (w*)
        optimal_w = self._solve_for_w_gpu(task_grads, avg_grad, phi)

        # Compute update vector (d)
        gw_optimal = optimal_w @ task_grads
        gw_norm = torch.norm(gw_optimal)

        if gw_norm.item() < 1e-8:
            update_vector = avg_grad
        else:
            regularization_term = (torch.sqrt(phi) / gw_norm) * gw_optimal
            update_vector = avg_grad + regularization_term

        # Update params
        lr = self.param_groups[0]['lr']
        pointer = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                num_param = p.numel()
                # Apply update: p.data = p.data - lr * d_slice
                p.data.add_(update_vector[pointer:pointer + num_param].view_as(p.data), alpha=-lr)
                pointer += num_param


    
        