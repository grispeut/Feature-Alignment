import torch
import torch.nn as nn
from utils.utils import compute_adv_loss


class PGD(nn.Module):
    def __init__(self, basic_model):
        super(PGD, self).__init__()
        self.basic_model = basic_model
        self.epsilon = 0.03

    def adv_sample_infer(self, input_img, targets, step_size, num_steps=1, random=False):
        x = input_img.clone().detach()
        if random:
            delta = torch.zeros_like(x).cuda()
            delta.uniform_(-self.epsilon, self.epsilon)
            x = x + delta
            x = torch.clamp(x, 0, 1)
        x.requires_grad_()
        for _ in range(num_steps):
            with torch.enable_grad():
                _, pred = self.basic_model(x)
                loss_adv, _ = compute_adv_loss(pred, targets, self.basic_model)
            x_grad = torch.autograd.grad(loss_adv, [x])[0]
            eta = step_size * torch.sign(x_grad)
            x.data = x.data + eta
            x.data = torch.min(torch.max(x.data, input_img - self.epsilon), input_img + self.epsilon)
            x.data = torch.clamp(x.data, 0, 1)

        return x.detach()

    def adv_sample_train(self, input_img, targets, step_size, all_bp=False, sf=0., num_steps=1, random=False):
        x = input_img.clone().detach()
        if random:
            delta = torch.zeros_like(x).cuda()
            delta.uniform_(-self.epsilon, self.epsilon)
            x = x + delta
            x = torch.clamp(x, 0, 1)
        x.requires_grad_()
        loss_clean = torch.zeros(1).cuda()
        ssfa_out = torch.zeros(1).cuda()
        for i in range(num_steps):

            if i == 0 and all_bp:
                pred, ssfa_out = self.basic_model(x, fa=True)
                loss_clean, _ = compute_adv_loss(pred, targets, self.basic_model)
                if sf > 0:
                    loss_clean = loss_clean * sf
                loss_clean.backward()
                eta = step_size * torch.sign(x.grad)
            else:
                pred = self.basic_model(x)
                loss_adv, _ = compute_adv_loss(pred, targets, self.basic_model)
                x_grad = torch.autograd.grad(loss_adv, [x], retain_graph=False)[0]
                eta = step_size * torch.sign(x_grad)

            x.data = x.data + eta
            x.data = torch.min(torch.max(x.data, input_img - self.epsilon), input_img + self.epsilon)
            x.data = torch.clamp(x.data, 0, 1)

        return x.detach(), ssfa_out.detach(), loss_clean
