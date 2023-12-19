import argparse
import shutil
from datetime import datetime
from importlib import import_module
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def load_from_config(cfg, **kwargs):
    module_str, cls_str = cfg["cls"].rsplit(".", 1)
    cls = getattr(import_module(module_str), cls_str)
    params = cfg.get("params", {})

    return cls(**params, **kwargs)


def run_model(*, network, model, model_cfg, dataset, ood_dataset, writer, out_dir, model_type):
    """Load checkpoint if available and run training if specified"""

    param_count = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Network has {param_count} trainable parameters.")

    if "checkpoint" in model_cfg:
        checkpoint = torch.load(model_cfg["checkpoint"])
        network.load_state_dict(checkpoint["network"])
        model.global_step = checkpoint["global_step"]
        print(f"Loaded checkpoint from {model_cfg['checkpoint']} "
                f"at global step {model.global_step}.")

    if "training" in model_cfg:
        # Set up optimizer
        opt = load_from_config(model_cfg["training"]["optimizer"], params=network.parameters())
        if "checkpoint" in model_cfg:
            opt.load_state_dict(checkpoint["optimizer"])

        # Set up dataloader
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=model_cfg["training"]["batch_size"])

        if "ood_dataset" in cfg:
            ood_loader = torch.utils.data.DataLoader(
                ood_dataset, batch_size=model_cfg["training"]["batch_size"])
        else:
            ood_loader = None

        # Set up metrics and tensorboard writing via callbacks
        callback_kwargs = {
            "dataset": dataset,
            "loader": loader,
            "ood_dataset": ood_dataset,
            "ood_loader": ood_loader,
            "writer": writer,
            "model": model,
            "network": network,
            "optimizer": opt,
            "output_dir": out_dir,
            "model_type": model_type
        }
        train_callbacks = [
            load_from_config(callback_cfg, **callback_kwargs)
            for callback_cfg in model_cfg["training"]["callbacks"]
        ]

        # Delete config options that aren't arguments for `model.train`
        del model_cfg["training"]["optimizer"]
        del model_cfg["training"]["batch_size"]
        del model_cfg["training"]["callbacks"]

        model.train(
            optim=opt,
            dataloader=loader,
            callbacks=train_callbacks,
            tqdm_level="batch",
            **model_cfg["training"]
        )

    if "evaluation" in model_cfg:
        # Set up dataloader
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=model_cfg["evaluation"]["batch_size"])

        if "ood_dataset" in cfg:
            ood_loader = torch.utils.data.DataLoader(
                ood_dataset, batch_size=model_cfg["evaluation"]["batch_size"])
        else:
            ood_loader = None

        # Set up metrics and tensorboard writing via callbacks
        callback_kwargs = {
            "dataset": dataset,
            "loader": loader,
            "ood_dataset": ood_dataset,
            "ood_loader": ood_loader,
            "writer": writer,
            "model": model,
            "network": network,
            "output_dir": out_dir,
            "model_type": model_type
        }
        eval_callbacks = [
            load_from_config(callback_cfg, **callback_kwargs)
            for callback_cfg in model_cfg["evaluation"]["callbacks"]
        ]

        # Callbacks that require args other than global step (ie. those meant for training)
        # will not work here
        for cb in eval_callbacks:
            cb.call(global_step=model.global_step)


def main(cfg):
    # Create subdirectory for model
    time_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    basename = f"{time_now}-{cfg['name']}" if cfg.get("name", None) else time_now
    out_dir = Path(cfg["output_root"]) / basename
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir)
    print(f"Created model directory {out_dir}")

    device = torch.device("cuda")

    # TensorBoard
    writer = SummaryWriter(log_dir=out_dir)
    writer.add_text("config", "```\n" + yaml.dump(cfg) + "```") # Log config verbatim as yaml

    # Datasets
    dataset = load_from_config(cfg["dataset"], transform=transforms.ToTensor())
    if "ood_dataset" in cfg:
        ood_dataset = load_from_config(cfg["ood_dataset"], transform=transforms.ToTensor())
    else:
        ood_dataset = None

    if "ebm" in cfg:
        ebm_cfg = cfg["ebm"]
        energy = load_from_config(ebm_cfg["energy"])
        ebm = load_from_config(ebm_cfg, energy=energy, device=device)

        run_model(
            network=energy,
            model=ebm,
            model_cfg=ebm_cfg,
            dataset=dataset,
            ood_dataset=ood_dataset,
            writer=writer,
            out_dir=out_dir,
            model_type="ebm",
        )

    if "implicit_manifold" in cfg:
        implicit_cfg = cfg["implicit_manifold"]
        mdf = load_from_config(implicit_cfg["mdf"])
        manifold = load_from_config(implicit_cfg, mdf=mdf, device=device)

        run_model(
            network=mdf,
            model=manifold,
            model_cfg=implicit_cfg,
            dataset=dataset,
            ood_dataset=ood_dataset,
            writer=writer,
            out_dir=out_dir,
            model_type="implicit_manifold",
        )

    if "constrained_ebm" in cfg:
        # Build and run a constrained EBM using the manifold instantiated above
        cebm_cfg = cfg["constrained_ebm"]
        energy = load_from_config(cebm_cfg["energy"])
        cebm = load_from_config(cebm_cfg, manifold=manifold, energy=energy, device=device)

        run_model(
            network=energy,
            model=cebm,
            model_cfg=cebm_cfg,
            dataset=dataset,
            ood_dataset=ood_dataset,
            writer=writer,
            out_dir=out_dir,
            model_type="constrained_ebm",
        )

    if "autoencoder" in cfg:
        ae_cfg = cfg["autoencoder"]
        encoder = load_from_config(ae_cfg["encoder"])
        decoder = load_from_config(ae_cfg["decoder"])
        network = torch.nn.Sequential(encoder, decoder)
        autoencoder = load_from_config(
            ae_cfg, encoder=encoder, decoder=decoder, device=device)

        run_model(
            network=network,
            model=autoencoder,
            model_cfg=ae_cfg,
            dataset=dataset,
            ood_dataset=ood_dataset,
            writer=writer,
            out_dir=out_dir,
            model_type="autoencoder",
        )

    if "pushforward_ebm" in cfg:
        pebm_cfg = cfg["pushforward_ebm"]
        energy = load_from_config(pebm_cfg["energy"])
        pebm = load_from_config(pebm_cfg, autoencoder=autoencoder, energy=energy, device=device)

        run_model(
            network=energy,
            model=pebm,
            model_cfg=pebm_cfg,
            dataset=dataset,
            ood_dataset=ood_dataset,
            writer=writer,
            out_dir=out_dir,
            model_type="pushforward_ebm",
        )


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    config_path = args.config_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
