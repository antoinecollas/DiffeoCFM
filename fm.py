import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torchdiffeq import odeint
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

from data import FastDataloader
from gaussian import ClassConditionalGaussianPrior
from model import MLP
from diffeo import DiffeomorphicMixin


class DiffeoCFM(DiffeomorphicMixin):
    def __init__(self, config):
        self.config = dict(config)
        diffeomorphism = self.config.get("DIFFEO")
        fm_types = {
            "classic": ConditionalFlowMatcher,
            "ot": ExactOptimalTransportConditionalFlowMatcher,
            "schrodinger_bridge": SchrodingerBridgeConditionalFlowMatcher,
            "variance_preserving": VariancePreservingConditionalFlowMatcher,
        }
        self.fm = fm_types[self.config["FM_TYPE"]]()
        super().__init__(diffeomorphism=diffeomorphism)
        self._prior = ClassConditionalGaussianPrior(random_state=self.config["RNG"])

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
        X = self._prior.sample(y_cond.cpu().numpy())
        X = torch.from_numpy(X).to(torch.float64).to(DEVICE)
        X = X.squeeze(0)
        return X

    def _time_sampler(self, bs):
        DEVICE = self.config["DEVICE"]
        return torch.rand(bs, dtype=torch.float64, device=DEVICE)

    def set_diffeomorphism(self, diffeomorphism: str | None) -> None:
        self.config["DIFFEO"] = diffeomorphism
        super().set_diffeomorphism(diffeomorphism)
        self._prior = ClassConditionalGaussianPrior(random_state=self.config["RNG"])

    def fit(self, X, y):
        config = self.config
        DEVICE = config["DEVICE"]
        EPOCHS = config["EPOCHS"]
        HIDDEN_DIM = config["HIDDEN_DIM"]
        BATCH_SIZE = config["BATCH_SIZE"]
        PRINT_EVERY = config["PRINT_EVERY"]
        RNG = config["RNG"]

        print("Training with DiffeoCFM.")

        X = np.asarray(X)
        y = np.asarray(y)

        # Split data into train and validation sets with stratification
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=RNG, shuffle=True
        )

        # Embed data into the working vector space
        X_train = self._fit_transform_features(X_train_raw)
        X_val = self._transform_features(X_val_raw)

        # Fit the prior in the same feature space
        self._prior.fit(X_train, y_train)

        # Dimensions
        n_features = X_train.shape[1]
        self.n_classes = len(np.unique(y_train))

        # Convert data to torch tensors
        X_train = torch.from_numpy(X_train).to(torch.float64).to(DEVICE)
        X_val = torch.from_numpy(X_val).to(torch.float64).to(DEVICE)
        y_train = torch.from_numpy(y_train).to(torch.long).to(DEVICE)
        y_val = torch.from_numpy(y_val).to(torch.long).to(DEVICE)

        vf = MLP(
            input_dim=n_features,
            cond_dim=self.n_classes,
            hidden_dim=HIDDEN_DIM,
            dtype=torch.float64,
        ).to(DEVICE)

        print(f"Vector field has {sum(p.numel() for p in vf.parameters())} parameters.")

        optimizer, scheduler = self._init_optim(vf)
        loss_fct = torch.nn.MSELoss(reduction="mean")

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

            for _, x0, x1, y1 in train_loader:
                x0, x1, y1 = x0.to(DEVICE), x1.to(DEVICE), y1.to(DEVICE)
                optimizer.zero_grad()
                t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1)
                y1 = F.one_hot(y1, num_classes=self.n_classes)
                loss = loss_fct(vf(xt, y1, t), ut)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            with torch.no_grad():
                for _, x0, x1, y1 in val_loader:
                    x0, x1, y1 = x0.to(DEVICE), x1.to(DEVICE), y1.to(DEVICE)
                    t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1)
                    y1 = F.one_hot(y1, num_classes=self.n_classes)
                    loss = loss_fct(vf(xt, y1, t), ut)
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

        self.vf = vf

        training_info = {
            "train_loss": np.array(train_losses_epoch),
            "val_loss": np.array(val_losses_epoch),
            "training_time": np.array(train_time),
        }
        return training_info

    def sample(self, y_cond):
        config = self.config
        DEVICE = config["DEVICE"]
        T_GRID = config["T_GRID"]

        class CondVFWrapper(torch.nn.Module):
            def __init__(self, vf, y_cond):
                super().__init__()
                self.vf = vf
                self.y_cond = y_cond

            def forward(self, x, t):
                return self.vf(x, self.y_cond, t)

        # Sample from the prior
        y_cond = torch.from_numpy(y_cond).to(torch.long).to(DEVICE)
        x_init = self._prior_sample_torch(y_cond)
        y_cond_oh = F.one_hot(y_cond, num_classes=self.n_classes)
        cond_vf = CondVFWrapper(self.vf, y_cond_oh).to(DEVICE)
        time_grid = self.config["T_GRID"]
        if not torch.is_tensor(time_grid):
            time_grid = torch.tensor(time_grid, dtype=torch.float64, device=DEVICE)
        else:
            time_grid = time_grid.to(device=DEVICE, dtype=torch.float64)

        def ode_rhs(t, x):
            return cond_vf(x, t)

        sol = odeint(
            ode_rhs,
            x_init,
            t=time_grid,
            method="dopri5",
            atol=1e-5,
            rtol=1e-5,
            options={"min_step": 1e-5},
        )

        sol_np = sol.detach().cpu().numpy()
        return self._inverse_transform_features(sol_np)
