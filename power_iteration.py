import torch
from torch import autograd
import torch.nn as nn
class PowerSVD(autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, p_buffer, q_buffer, iter):
        for i in range(iter):
            if i == iter - 1:
                p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
            q_buffer[0] = input @ p_buffer[0]
            if i == iter - 1:
                q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
            # p_buffer[0] = input.permute((0, 1, 3, 2)) @ q_buffer[0]
            p_buffer[0] = torch.transpose(input,0,1) @ q_buffer[0]
        ctx.p_buffer, ctx.q_buffer = p_buffer, q_buffer
        ctx.iter = iter
        # return q_buffer[0] @ p_buffer[0].permute((0, 1, 3, 2))
        return q_buffer[0] @ torch.transpose(p_buffer[0],0,1)
    

    @staticmethod
    def backward(ctx, grad_output):
        iter = ctx.iter
        p_buffer, q_buffer = ctx.p_buffer, ctx.q_buffer
        for i in range(iter):
            if i == iter - 1:
                p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
            q_buffer[0] = grad_output @ p_buffer[0]
            if i == iter - 1:
                q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
            # p_buffer[0] = grad_output.permute((0, 1, 3, 2)) @ q_buffer[0]
            p_buffer[0] = torch.transpose(grad_output, 0 ,1) @ q_buffer[0]
        # return q_buffer[0] @ p_buffer[0].permute((0, 1, 3, 2)), None, None, None
        return q_buffer[0] @ torch.transpose(p_buffer[0],0,1), None, None, None


class PowerSVDLayer(nn.Module):
    def __init__(self, rank, shape, iter) -> None:
        super(PowerSVDLayer, self).__init__()
        # self.p_buffer = torch.nn.Parameter(
        #     torch.randn((int(shape[0]), int(shape[1]), int(shape[2]), rank))
        # )
        self.p_buffer = torch.nn.Parameter(
            torch.randn(( int(shape[0]), rank))
        )
        self.q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[1]), rank))
        )
        self.iter = iter

    def forward(self, input):
        return PowerSVD.apply(input, [self.p_buffer], [self.q_buffer], self.iter)
