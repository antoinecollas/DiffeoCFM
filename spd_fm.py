import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import jvp
from torchdiffeq import odeint
from sklearn.model_selection import train_test_split

from data import FastDataloader
from gaussian import ClassConditionalGaussianPrior
from model import MLP

from spd import SPD


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield, manifold):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold

    def forward(self, x, y, t):
        x = self.manifold.projx(x)
        v = self.vecfield(x, y, t)
        v = self.manifold.proju(x, v)
        v = self.manifold.metric_normalized(x, v)
        return v


class CondVFWrapper(torch.nn.Module):
    def __init__(self, vf, y_cond):
        super().__init__()
        self.vf = vf
        self.y_cond = y_cond

    def forward(self, x, t):
        return self.vf(x, self.y_cond, t)


class SPDConditionalFlowMatching:
    def __init__(self, config):
        self.config = config
        self._prior = ClassConditionalGaussianPrior(random_state=config["RNG"])
        self.manifold = SPD()

    def _init_optim(self, model):
        config = self.config
        LR = config["LR"]
        FACTOR_LR = config["FACTOR_LR"]
        WARMUP_EPOCHS = config["WARMUP_EPOCHS"]
        EPOCHS = config["EPOCHS"]

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=FACTOR_LR,
                    end_factor=1.0,
                    total_iters=WARMUP_EPOCHS,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * FACTOR_LR
                ),
            ],
            milestones=[WARMUP_EPOCHS],
        )
        return optimizer, scheduler

    def _prior_sample_torch(self, y_cond):
        DEVICE = self.config["DEVICE"]

        # Sample from the prior (assumed to be in the tangent space at the eye)
        X = self._prior.sample(y_cond.cpu().numpy())
        X = torch.from_numpy(X).to(torch.float64).to(DEVICE)

        # Expmap at the eye
        eye = torch.eye(self.dim, dtype=torch.float64).to(DEVICE)
        eye = self.manifold.vectorize(eye)
        X = self.manifold.expmap(eye, X)

        X = X.squeeze(0)
        return X

    def _time_sampler(self, bs):
        DEVICE = self.config["DEVICE"]
        return torch.rand(bs, dtype=torch.float64, device=DEVICE)

    def rfm_loss_fn(self, x0, x1, y1, t):
        def SPD_geodesic(t):
            return self.manifold.geodesic(x0, x1, t)

        xt, ut = jvp(SPD_geodesic, (t,), (torch.ones_like(t).to(t),))

        diff = self.vf(xt, y1, t) - ut
        return self.manifold.inner(xt, diff, diff).mean() / self.dim

    def fit(self, X, y):
        config = self.config
        DEVICE = config["DEVICE"]
        EPOCHS = config["EPOCHS"]
        HIDDEN_DIM = config["HIDDEN_DIM"]
        BATCH_SIZE = config["BATCH_SIZE"]
        PRINT_EVERY = config["PRINT_EVERY"]
        RNG = config["RNG"]

        print("Training with SPD-CFM.")

        self.dim = X.shape[1]
        man = self.manifold

        # Vectorize data
        X = man.vectorize(X)

        # Split data into train and validation sets with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=RNG, shuffle=True
        )

        # Convert data to torch tensors
        X_train = torch.from_numpy(X_train).to(torch.float64).to(DEVICE)
        X_val = torch.from_numpy(X_val).to(torch.float64).to(DEVICE)
        y_train = torch.from_numpy(y_train).to(torch.long).to(DEVICE)
        y_val = torch.from_numpy(y_val).to(torch.long).to(DEVICE)

        # Fit the prior
        # 1) apply matrix Riemannian logarithm at the eye (i.e., log(X))
        eye = torch.eye(self.dim, dtype=torch.float64).to(DEVICE)
        eye = man.vectorize(eye)
        X_train_log = man.logmap(eye, X_train)
        # 2) fit the prior on the log vectors
        self._prior.fit(X_train_log.cpu().numpy(), y_train.cpu().numpy())

        # Dimensions
        n_features = X_train.shape[1]
        self.n_classes = len(np.unique(y_train.cpu().numpy()))

        self.vf = ProjectToTangent(
            MLP(
                input_dim=n_features,
                cond_dim=self.n_classes,
                hidden_dim=HIDDEN_DIM,
                dtype=torch.float64,
            ),
            manifold=self.manifold,
        ).to(DEVICE)

        print(
            f"Vector field has {sum(p.numel() for p in self.vf.parameters())} parameters."
        )

        optimizer, scheduler = self._init_optim(self.vf)

        train_losses_epoch, val_losses_epoch = [], []
        train_time = []

        train_loader = FastDataloader(
            x1=X_train,
            y1=y_train,
            time_sampler=self._time_sampler,
            prior=self._prior_sample_torch,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )
        val_loader = FastDataloader(
            x1=X_val,
            y1=y_val,
            time_sampler=self._time_sampler,
            prior=self._prior_sample_torch,
            batch_size=len(X_val),
            shuffle=False,
            drop_last=True,
        )

        for epoch in range(EPOCHS):
            train_losses, val_losses = [], []
            start = time.time()

            for t, x0, x1, y1 in train_loader:
                x0, x1, y1 = x0.to(DEVICE), x1.to(DEVICE), y1.to(DEVICE)
                optimizer.zero_grad()
                y1 = F.one_hot(y1, num_classes=self.n_classes)
                loss = self.rfm_loss_fn(x0, x1, y1, t)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            with torch.no_grad():
                for t, x0, x1, y1 in val_loader:
                    x0, x1, y1 = x0.to(DEVICE), x1.to(DEVICE), y1.to(DEVICE)
                    y1 = F.one_hot(y1, num_classes=self.n_classes)
                    loss = self.rfm_loss_fn(x0, x1, y1, t)
                    val_losses.append(loss.item())

            scheduler.step()

            train_loss_mean = np.mean(train_losses)
            val_loss_mean = np.mean(val_losses)
            train_losses_epoch.append(train_loss_mean)
            val_losses_epoch.append(val_loss_mean)
            train_time.append(time.time() - start)

            if epoch % PRINT_EVERY == 0 or epoch == 0 or epoch == EPOCHS - 1:
                print(
                    f"| epoch {epoch:3d} | time {np.sum(np.array(train_time)[-PRINT_EVERY:]):.2f}s "
                    f"| lr {scheduler.get_last_lr()[0]:.2e} "
                    f"| loss {train_loss_mean:.2e} | val loss {val_loss_mean:.2e} |"
                )

        training_info = {
            "train_loss": np.array(train_losses_epoch),
            "val_loss": np.array(val_losses_epoch),
            "training_time": np.array(train_time),
        }
        return training_info

    def sample(self, y_cond):
        DEVICE = self.config["DEVICE"]

        man = self.manifold

        y_cond = torch.from_numpy(y_cond).to(torch.long).to(DEVICE)
        x0 = self._prior_sample_torch(y_cond).to(torch.float64)

        # one-hot labels for conditioning
        y_onehot = F.one_hot(y_cond, num_classes=self.n_classes)
        vf_cond = CondVFWrapper(self.vf, y_onehot).to(DEVICE)

        # rhs needs signature  (t, x) -> dx/dt  for torchdiffeq
        def vf_cond_(t, x):
            return vf_cond(x, t)

        # integrate from t=0 to t=1 with dopri5, tol = 1e-5
        with torch.no_grad():
            try:
                x1 = odeint(
                    vf_cond_,
                    x0,
                    t=torch.linspace(0, 1, 2).to(DEVICE),
                    method="dopri5",
                    atol=1e-5,
                    rtol=1e-5,
                    options={"min_step": 1e-5},
                )
            except AssertionError:
                # In case of an error, we return the initial point
                x1 = x0[np.newaxis, ...]

        # Project the output on the manifold
        x1 = man.projx(x1)

        # back to (d, d) SPD matrices and numpy
        x1 = man.devectorize(x1)
        x1 = x1.cpu().numpy()

        return x1
