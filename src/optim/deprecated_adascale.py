import random
import math

import torch
import torch.optim


class AdaScale(object):
    def __init__(self, optimizer, scale=1, adam=True):

        self.optimizer = optimizer
        self.scale = scale
        self.adam = adam
        self._step = 0

        num_params = sum(len(pg["params"]) for pg in optimizer.param_groups)

        self.smoothing = 1 - scale / 1000#0.997
        self.eps = 1e-6
        self.local_grads = {}
        self.mean = 0
        self.var = 0
        self.gain = 1

    def step(self):
        self._step += 1
        self.accumulate()

        if self._step % self.scale == 0:
            step2 = self._step / self.scale
            if self.scale > 1:
                denominator, numerator = 0, 0
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        p.grad.div_(self.scale)
                        grad = p.grad.clone()

                        if self.adam and 'exp_avg_sq' in self.optimizer.state[p] and step2 > 10:
                            state = self.optimizer.state[p]
                            exp_avg_sq = state['exp_avg_sq']
                            beta2 = group['betas'][1]
                            correction = 1 - beta2 ** state['step']
                            pinv = (exp_avg_sq.sqrt() / math.sqrt(correction)).add_(group['eps'])
                        else:
                            pinv = 1

                        denominator += (grad / pinv).pow(2).sum()
                        for g in self.local_grads[p]:
                            numerator += (g / pinv).pow(2).sum()

                numerator = numerator / self.scale

                # print('\n', denominator.item(), numerator.item(), numerator.item()/denominator.item())

                mean = (self.scale * denominator - numerator) / (self.scale - 1)
                var = (numerator - denominator) * self.scale / (self.scale - 1)
                mean = mean if mean > 0 else 0
                var = var if var > self.eps else self.eps
                if step2 > 1 / (1 - self.smoothing):
                    self.mean = self.smoothing * self.mean + (1 - self.smoothing) * mean
                    self.var = self.smoothing * self.var + (1 - self.smoothing) * var
                else:
                    self.mean = (self.mean * (step2 - 1) + mean) / step2
                    self.var = (self.var * (step2 - 1) + var) / step2

                bias_correction = 1 - self.smoothing ** step2
                mean_final = self.mean # / bias_correction
                var_final = self.var # / bias_correction
                gain = (mean_final + var_final) / (mean_final + var_final / self.scale)
                self.gain = float(gain.item())
            else:
                gain = 1
            init_lr = [pg["lr"] for pg in self.optimizer.param_groups]

            for group in self.optimizer.param_groups:
                group["lr"] = gain * group["lr"]

            self.optimizer.step()

            # reset
            self.local_grads = {}
            for lr, param_group in zip(init_lr, self.optimizer.param_groups):
                param_group["lr"] = lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def accumulate(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.clone()
                if p not in self.local_grads:
                    self.local_grads[p] = [grad]
                else:
                    prev_grad = torch.stack(self.local_grads[p]).sum(0)
                    self.local_grads[p].append(grad - prev_grad)


class EfficientAdaScale(object):
    def __init__(self, optimizer, scale=1, adam=True):

        self.optimizer = optimizer
        self.scale = scale
        self.adam = adam
        self._step = 0

        num_params = sum(len(pg["params"]) for pg in optimizer.param_groups)

        self.smoothing = 1 - scale / 1000
        self.eps = 1e-6
        self.local_grad_sq = 0
        self.prev_grad = {}
        self.mean = 0
        self.var = 0
        self.gain = 1
        self.warmup = 10

    def step(self):
        self._step += 1

        if self.scale == 1:
            self.optimizer.step()
        else:
            self.accumulate()

            # perform optimizer.step()
            if self._step % self.scale == 0:
                step2 = self._step / self.scale
                global_grad_sq = 0
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        p.grad.div_(self.scale)
                        pinv = self.get_preconditioner_inv(p, group)
                        grad = p.grad.clone()
                        global_grad_sq += (grad / pinv).pow(2).sum()

                local_grad_sq = self.local_grad_sq / self.scale

                # print('\n', global_grad_sq.item(), local_grad_sq.item(), global_grad_sq.item() / local_grad_sq.item())

                mean = (self.scale * global_grad_sq - local_grad_sq) / (self.scale - 1)
                var = (local_grad_sq - global_grad_sq) * self.scale / (self.scale - 1)
                mean = mean if mean > 0 else 0
                var = var if var > self.eps else self.eps
                if step2 > 1 / (1 - self.smoothing):
                    self.mean = self.smoothing * self.mean + (1 - self.smoothing) * mean
                    self.var = self.smoothing * self.var + (1 - self.smoothing) * var
                else:
                    self.mean = (self.mean * (step2 - 1) + mean) / (step2)
                    self.var = (self.var * (step2 - 1) + var) / step2

                mean_final = self.mean
                var_final = self.var
                gain = (mean_final + var_final) / (mean_final + var_final / self.scale)
                self.gain = float(gain.item())
                init_lr = [pg["lr"] for pg in self.optimizer.param_groups]

                for group in self.optimizer.param_groups:
                    group["lr"] = gain * group["lr"]

                self.optimizer.step()

                # reset
                self.local_grad_sq = 0
                self.prev_grad = {}
                for lr, param_group in zip(init_lr, self.optimizer.param_groups):
                    param_group["lr"] = lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def accumulate(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.clone()
                pinv = self.get_preconditioner_inv(p, group)
                if p not in self.prev_grad:
                    self.local_grad_sq += (grad / pinv).pow(2).sum()
                else:
                    self.local_grad_sq += ((grad - self.prev_grad[p]) / pinv).pow(2).sum()
                self.prev_grad[p] = grad

    def get_preconditioner_inv(self, param, group):
        if not self.adam:
            return 1
        else:
            state = self.optimizer.state[param]
            if 'exp_avg_sq' not in state or state['step'] < self.warmup:
                return 1
            else:
                exp_avg_sq = state['exp_avg_sq']
                beta2 = group['betas'][1]
                bias_correction = 1 - beta2 ** state['step']
                pinv = (exp_avg_sq.sqrt() / math.sqrt(bias_correction)).add_(group['eps'])
                return pinv
