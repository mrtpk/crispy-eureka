from collections import defaultdict
from keras.callbacks import Callback

class WRWDScheduler(Callback):
    """Warm restart scheduler for optimizers with decoupled weight decay
    
    This Keras callback should be used with TensorFlow optimizers 
    with decoupled weight decay, such as tf.contrib.opt.AdamWOptimizer
    or tf.contrib.opt.MomentumWOptimizer. Warm restarts include 
    cosine annealing with periodic restarts for both learning rate 
    and weight decay. Normalized weight decay is also included.
    
    # Example
    ```python
    lr = 0.001
    wd = 0.01

    optimizer = tf.contrib.opt.AdamWOptimizer(
        learning_rate=lr,
        weight_decay=wd)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    cb_wrwd = WRWDScheduler(steps_per_epoch=100, lr=lr, wd_norm=wd)

    model.fit(x_train, y_train, callbacks=[cb_wrwd])

    figure(1);plt.plot(cb_wrwd.history['lr'])
    figure(2);plt.plot(cb_wrwd.history['wd'])
    ```

    # Arguments
        steps_per_epoch: number of training batches per epoch
        lr: initial learning rate
        wd_norm: normalized weight decay
        eta_min: minimum of the multiplier
        eta_max: maximum of the multiplier
        eta_decay: decay rate of eta_min/eta_max after each restart
        cycle_length: number of epochs in the first restart cycle
        cycle_mult_factor: rate to increase the number of epochs 
            in a cycle after each restart

    # Reference
        arxiv.org/abs/1608.03983
        arxiv.org/abs/1711.05101
        jeremyjordan.me/nn-learning-rate
    """
    
    def __init__(self,
                 steps_per_epoch,
                 lr=0.001,
                 wd_norm=0.03,
                 eta_min=0.0,
                 eta_max=1.0,
                 eta_decay=1.0,
                 cycle_length=10,
                 cycle_mult_factor=1.5):
        
        self.lr = lr
        self.wd_norm = wd_norm

        self.steps_per_epoch = steps_per_epoch

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_decay = eta_decay

        self.steps_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.cycle_mult_factor = cycle_mult_factor

        self.wd = wd_norm / (steps_per_epoch*cycle_length)**0.5

        self.history = defaultdict(list)

    def cal_eta(self):
        '''Calculate eta'''
        fraction_to_restart = self.steps_since_restart / (self.steps_per_epoch * self.cycle_length)
        eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + np.cos(fraction_to_restart * np.pi))
        return eta

    def on_train_batch_begin(self, batch, logs={}):
        '''update learning rate and weight decay'''
        eta = self.cal_eta()
        self.model.optimizer.optimizer._learning_rate = eta * self.lr
        self.model.optimizer.optimizer._weight_decay = eta * self.wd

    def on_train_batch_end(self, batch, logs={}):
        '''Record previous batch statistics'''
        logs = logs or {}
        self.history['lr'].append(self.model.optimizer.optimizer._learning_rate)
        self.history['wd'].append(self.model.optimizer.optimizer._weight_decay)
        for k, v in logs.items():
            self.history[k].append(v)

        self.steps_since_restart += 1

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary'''
        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.cycle_mult_factor)
            self.next_restart += self.cycle_length
            self.eta_min *= self.eta_decay
            self.eta_max *= self.eta_decay
            self.wd = self.wd_norm / (self.steps_per_epoch*self.cycle_length)**0.5