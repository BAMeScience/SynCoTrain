import os

import pandas as pd
import torch
from torch.optim.lr_scheduler import CyclicLR

from crabnet.utils.utils import (Lamb, Lookahead, RobustL1, BCEWithLogitsLoss,
                         EDM_CsvLoader, Scaler, DummyScaler, count_parameters)
from crabnet.utils.get_compute_device import get_compute_device
from crabnet.utils.optim import SWA

from syncotrain.lib import puLearning
from syncotrain.src import configuration
from syncotrain.lib.classifier import Classifier

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


output_dir = None


def get_config(name, train_ratio, val_ratio, test_ratio):
    config = loadjson('syncotrain/lib/alignn/default_class_config.json')
    config["output_dir"] = configuration.result_dir + "/tmp"

    classifier = configuration.config[name]
    for key in classifier:
        typ = type(config[f"{key}"]).__name__
        if typ == "int":
            value = int(classifier[f"{key}"])
        elif typ == "float":
            value = float(classifier[f"{key}"])
        elif typ == "bool":
            value = bool(classifier[f"{key}"])
        else:
            value = classifier[f"{key}"]
        config[f"{key}"] = value

    config["train_ratio"] = train_ratio
    config["val_ratio"] = val_ratio
    config["test_ratio"] = test_ratio
    config["write_checkpoint"] = True
    (configuration.project_path / config["output_dir"]).mkdir(parents=True, exist_ok=True)
    global output_dir
    output_dir = config["output_dir"]
    config = TrainingConfig(**config)
    return config


class Crabnet(Classifier):
    def __init__(self, name):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._name = name
        self._model = None

    def fit(self, X: pd.Series, y: pd.Series):
        def fit(self, epochs=None, checkin=None, losscurve=False):
        assert_train_str = 'Please Load Training Data (self.train_loader)'
        assert_val_str = 'Please Load Validation Data (self.data_loader)'
        assert self.train_loader is not None, assert_train_str
        assert self.data_loader is not None, assert_val_str
        self.loss_curve = {}
        self.loss_curve['train'] = []
        self.loss_curve['val'] = []

        # change epochs_step
        # self.epochs_step = 10
        self.epochs_step = 1
        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'stepping every {self.step_size} training passes,',
              f'cycling lr every {self.epochs_step} epochs')
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            print(f'running for {epochs} epochs')
        if checkin is None:
            checkin = self.epochs_step * 2
            print(f'checkin at {self.epochs_step*2} '
                  f'epochs to match lr scheduler')
        if epochs % (self.epochs_step * 2) != 0:
            # updated_epochs = epochs - epochs % (self.epochs_step * 2)
            # print(f'epochs not divisible by {self.epochs_step * 2}, '
            #       f'updating epochs to {updated_epochs} for learning')
            updated_epochs = epochs
            epochs = updated_epochs

        self.step_count = 0
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)

        lr_scheduler = CyclicLR(self.optimizer,
                                base_lr=1e-4,
                                max_lr=6e-3,
                                cycle_momentum=False,
                                step_size_up=self.step_size)

        self.swa_start = 2  # start at (n/2) cycle (lr minimum)
        self.lr_scheduler = lr_scheduler
        self.stepping = True
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        self.discard_n = 3

        for epoch in range(epochs):
            self.epoch = epoch
            self.epochs = epochs
            ti = time()
            self.train()
            # print(f'epoch time: {(time() - ti):0.3f}')
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])

            ##################################
            # Force evaluate dataset so that we can capture it in the hook
            # here we are using the train_loader, but we can also use
            # general data_loader
            if self.capture_every == 'epoch':
                # print('capturing every epoch!')
                # print(f'data_loader size: {len(self.data_loader.dataset)}')
                self.capture_flag = True
                # (act, pred, formulae, uncert)
                self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                self.capture_flag = False
            ##################################

            if (epoch+1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                ti = time()
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                dt = time() - ti
                datasize = len(act_t)
                # print(f'inference speed: {datasize/dt:0.3f}')
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)
                epoch_str = f'Epoch: {epoch}/{epochs} ---'
                train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
                val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f'train auc: {train_auc:0.3f}'
                    val_str = f'val auc: {val_auc:0.3f}'
                print(epoch_str, train_str, val_str)

                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    if (self.epoch+1) % (self.epochs_step * 2) == 0:
                        self.xswa.append(self.epoch)
                        self.yswa.append(mae_v)

                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                    xval[0] = 0
                    plt.plot(xval, self.loss_curve['train'],
                             'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'],
                             's--', label='val_mae')
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                    plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.show()

            if (epoch == epochs-1 or
                self.optimizer.discard_count >= self.discard_n):
                # save output df for stats tracking
                xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                xval[0] = 0
                tval = self.loss_curve['train']
                vval = self.loss_curve['val']
                os.makedirs('figures/lc_data', exist_ok=True)
                df_loss = pd.DataFrame([xval, tval, vval]).T
                df_loss.columns = ['epoch', 'train loss', 'val loss']
                df_loss['swa'] = ['n'] * len(xval)
                df_loss.loc[df_loss['epoch'].isin(self.xswa), 'swa'] = 'y'
                df_loss.to_csv(f'figures/lc_data/{self.model_name}_lc.csv',
                               index=False)

                # save output learning curve plot
                plt.figure(figsize=(8, 5))
                xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                xval[0] = 0
                plt.plot(xval, self.loss_curve['train'],
                         'o-', label='train_mae')
                plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                plt.title(f'{self.model_name}')
                plt.xlabel('epochs')
                plt.ylabel('MAE')
                plt.legend()
                plt.savefig(f'figures/lc_data/{self.model_name}_lc.png')

            if self.optimizer.discard_count >= self.discard_n:
                print(f'Discarded: {self.optimizer.discard_count}/'
                      f'{self.discard_n} weight updates, '
                      f'early-stopping now 🙅🛑')
                self.optimizer.swap_swa_sgd()
                break

        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()

    def predict(self, X: pd.Series):
        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0])/2)
        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=list)
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data
                if self.capture_flag:
                    self.formula_current = None
                    # HACK for PyTorch v1.8.0
                    # this output used to be a list, but is now a tuple
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()
                src, frac = X.squeeze(-1).chunk(2, dim=1)
                src = src.to(self.compute_device,
                             dtype=torch.long,
                             non_blocking=True)
                frac = frac.to(self.compute_device,
                               dtype=data_type_torch,
                               non_blocking=True)
                y = y.to(self.compute_device,
                         dtype=data_type_torch,
                         non_blocking=True)
                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                uncertainty = torch.exp(uncertainty) * self.scaler.std
                prediction = self.scaler.unscale(prediction)
                if self.classification:
                    prediction = torch.sigmoid(prediction)

                data_loc = slice(i*self.batch_size,
                                 i*self.batch_size+len(y),
                                 1)

                atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().detach().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy().astype('float32')
                formulae[data_loc] = formula
        self.model.train()

        return (act, pred, formulae, uncert)

        # return data['y']
