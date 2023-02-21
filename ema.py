import torch

class OptimizerEMA(object):
    '''
    EMA optimizer which can optionally apply EMA to BN statistics, with eman=True (see EMAN paper by Cai et al)
    '''
    def __init__(self, model, ema_model, alpha=0.999, eman=True, ramp_up=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.eman = eman
        self.step = 0
        self.ramp_up = ramp_up


    def update(self):
        if self.ramp_up:
            _alpha = min(self.alpha, (self.step + 1)/(self.step + 10)) 
        else:
            _alpha = self.alpha
        self.step += 1
        one_minus_alpha = 1.0 - _alpha

        # update learnable parameters
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.mul_(_alpha)
            ema_param.add_(param * one_minus_alpha)

        if self.eman:
            # update buffers (aka, non-learnable parameters). These are usually only BN stats
            for buffer, ema_buffer in zip(self.model.buffers(), self.ema_model.buffers()):
                if ema_buffer.dtype == torch.float32:      
                    ema_buffer.mul_(_alpha)
                    ema_buffer.add_(buffer * one_minus_alpha)