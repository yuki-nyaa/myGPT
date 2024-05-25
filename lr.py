import math

class BaseLRScheduler: pass

class CosineDecay(BaseLRScheduler):
    def __init__(self, min_lr, max_lr, warmup_iters, decay_iters):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
    def __call__(self, iter_num):
        # 1) Linear warmup for warmup_iters steps
        if iter_num < self.warmup_iters:
            return self.max_lr * (iter_num+1) / self.warmup_iters
        # 2) If i`ter_num > lr_decay_iters`, return min learning rate.
        if iter_num > self.decay_iters:
            return self.min_lr
        # 3) In between, use cosine decay down to min learning rate.
        decay_ratio = (iter_num - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 1..0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    def __str__(self):
        return f"CosineDecay with `min_lr`={self.min_lr:e}, `max_lr`={self.max_lr:e}, `warmup_iters`={self.warmup_iters}, `decay_iters`={self.decay_iters}"
    def __eq__(self,other):
        return self.min_lr == other.min_lr \
            and self.max_lr == other.max_lr \
            and self.warmup_iters == other.warmup_iters \
            and self.decay_iters == other.decay_iters

class LinearDecay(BaseLRScheduler):
    def __init__(self, max_lr, min_lr, warmup_iters, decay_iters):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
    def __call__(self, iter_num):
        # 1) Linear warmup for warmup_iters steps
        if iter_num < self.warmup_iters:
            return self.max_lr * (iter_num+1) / self.warmup_iters
        # 2) If i`ter_num > lr_decay_iters`, return min learning rate.
        if iter_num > self.decay_iters:
            return self.min_lr
        # 3) In between, use cosine decay down to min learning rate.
        decay_ratio = (iter_num - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 1-decay_ratio # coeff ranges 1..0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    def __str__(self):
        return f"LinearDecay with `min_lr`={self.min_lr:e}, `max_lr`={self.max_lr:e}, `warmup_iters`={self.warmup_iters}, `decay_iters`={self.decay_iters}"
    def __eq__(self,other):
        return self.min_lr == other.min_lr \
            and self.max_lr == other.max_lr \
            and self.warmup_iters == other.warmup_iters \
            and self.decay_iters == other.decay_iters

class ConstLR(BaseLRScheduler):
    def __init__(self, lr):
        self.lr = lr
    def __call__(self, iter_num): return self.lr
    def __str__(self): return f"ConstLR with `lr`={self.lr:e}"
    def __eq__(self,other): return self.lr == other.lr
