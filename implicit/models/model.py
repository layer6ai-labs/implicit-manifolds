from abc import ABC, abstractmethod

from tqdm import tqdm


class Model(ABC):
    """Abstract base class for a 'model' with a `.train` method.

    By 'model,' we mean an object that  holds nn.Modules, training details,
    and sampling details for the model.
    """
    @abstractmethod
    def train_step(self, optim, batch, *args, **kwargs):
        """Run a single training step."""
        pass

    def train(self, optim, dataloader, epochs, *args, callbacks=[], tqdm_level="epoch", **kwargs):
        """Fit the model to the dataloader using its `model.train_step` method.

        Args:
            optim: an optimizer for the parameters of `self.encoder` and `self.decoder`
            dataloader: iterable from which to load training batches
            epochs: number of epochs to train
            callbacks: an iterable of callbacks to run at the end of every batch
            tqdm_level: level of training to show tqdm bar; must be in {"epoch", "batch", None}
            args, kwargs: arguments for the underlying model's `train_batch` method
        """

        if not hasattr(self, "global_step"):
            self.global_step = 0

        # Set up epoch progress bar
        assert tqdm_level in {"epoch", "batch", None}
        if tqdm_level == "epoch":
            epoch_iter = pbar = tqdm(range(epochs))
            batch_iter = dataloader
        else: # tqdm_level is either None or "batch"; set vars for None, and override every epoch
            epoch_iter = range(epochs)
            batch_iter = dataloader
            pbar = None

        # Training loop
        for _ in epoch_iter:

            # Need to set up new tqdm progress bar for epoch
            if tqdm_level == "batch":
                batch_iter = pbar = tqdm(dataloader)

            for batch in batch_iter:
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    batch, _ = batch

                stats = self.train_step(optim, batch, *args, **kwargs)
                self.global_step += 1

                for cb in callbacks:
                    cb(
                        global_step=self.global_step,
                        batch=batch,
                        stats=stats,
                        pbar=pbar,
                    )
