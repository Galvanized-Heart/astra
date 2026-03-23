# CAGrad-Fast implementation as a multi-task learning optimizer wrapper from Conflict-Averse Gradient Descent for 
# Multi-task Learning by Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, Qiang Liu in 2021. https://doi.org/10.48550/arXiv.2110.14048

import torch
import torch.nn.utils as utils

class CAGrad:
    def __init__(self, optimizer, c=0.5, lr_w=25, momentum_w=0.5, epochs_w=20):
        self.optimizer = optimizer
        self.c = c # TODO: Experiment with c = {0.1, 0.5, 0.9}
        self.lr_w = lr_w
        self.momentum_w = momentum_w
        self.epochs_w = epochs_w
        
    def _solve_update_vector_gpu(self, task_grads):
        """
        Solves objective function (F(w)) to obtain optimal w (w*) and compute optimal update vector (d*).
        """
        
        num_tasks = len(task_grads)
        avg_grad = torch.mean(task_grads, dim=0)
        phi = self.c**2 * (avg_grad @ avg_grad) # phi = c^2 * ||g_0||_2^2
        # NOTE: For 1D tensor, L2 norm = dot prod to tensor.
        
        # Precalculate Gram matrix (g_i^T g_j) and g_i^T g_0
        GG = task_grads @ task_grads.T
        Gg = (task_grads @ avg_grad).unsqueeze(1)

        w_unconstrained = torch.zeros(num_tasks, 1, device=task_grads.device, requires_grad=True)
        w_optimizer = torch.optim.SGD([w_unconstrained], lr=self.lr_w, momentum=self.momentum_w) # High lr to find minimum w faster

        # c*||g0|| = sqrt(phi)
        c_prime = torch.sqrt(phi) 

        # Perform optimization steps
        for _ in range(self.epochs_w):
            w_optimizer.zero_grad()

            # Apply softmax to get weights w on probability simplex
            w = torch.softmax(w_unconstrained, dim=0)

            # Calculate objective function F(w) using precalculated matrices
            gw_sq_norm = w.T @ GG @ w
            gw_g0 = w.T @ Gg
            objective = gw_g0 + c_prime * torch.sqrt(gw_sq_norm + 1e-8)

            objective.backward()
            w_optimizer.step()

        optimal_w = torch.softmax(w_unconstrained, dim=0).detach().view(-1)

        # Compute update vector (d)
        gw_optimal = optimal_w @ task_grads
        gw_norm = torch.linalg.norm(gw_optimal)
        
        if gw_norm.item() < 1e-8:
            return avg_grad
        else:
            return avg_grad + (torch.sqrt(phi) / gw_norm) * gw_optimal

    def step(self, task_losses: dict):
        """
        Performs a single step of CAGrad multi-task updates.
        """

        self.optimizer.zero_grad()
        
        params = self.optimizer.param_groups[0]['params']
        
        # Compute individual gradients
        grads = []
        valid_task_indices = sorted(task_losses.keys())
        num_tasks = len(valid_task_indices)
        
        for i in range(num_tasks):
            for p in params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            task_key = valid_task_indices[i]
            task_losses[task_key].backward(retain_graph=(i < num_tasks - 1)) # retain_graph for all but last
            grad_vec = utils.parameters_to_vector([p.grad.clone() for p in params if p.grad is not None])
            grads.append(grad_vec)

        if not grads: 
            return

        task_grads = torch.stack(grads) # [num_tasks, num_params]
        
        # Compute update direction (d*)
        update_vector = self._solve_update_vector_gpu(task_grads)
        
        # Set final gradient on the parameters
        utils.vector_to_parameters(update_vector, [p.grad for p in params])
        
        # Step internal optimizer
        self.optimizer.step()