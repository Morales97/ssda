import torch



class OptimizerEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())


    def update(self, step):
        _alpha = min(self.alpha, (step + 1)/(step + 10)) # ramp up EMA
        one_minus_alpha = 1.0 - _alpha

        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                # Update Exponential Moving Average parameters
                ema_param.mul_(_alpha)
                ema_param.add_(param * one_minus_alpha)