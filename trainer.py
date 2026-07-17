import torch
from pathlib import Path
import re
from tqdm import tqdm
from model.loss import Loss

class Trainer:
    def __init__(self,model, train_loader, config, lr = 3e-4, device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.config = config
        self.criterion = Loss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress = tqdm(
                self.train_loader,
                desc="Training",
            )
        for inputs, target in progress:
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0
            )
            self.optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader)

    def save_checkpoint(
        self,
        epoch,
        loss,
        path="checkpoints/gpt.pt",
    ):
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "loss": loss,
        }

        torch.save(checkpoint, path)

    def train(self, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            loss = self.train_epoch()
            print(
                f"Epoch { epoch + 1}:"
                f"Loss: {loss:.4f}"
            )
            self.save_checkpoint(
                epoch=epoch + 1,
                loss=loss,
                path=f"checkpoints/gpt_{epoch+1}.pt",
            )

    def load_checkpoint(self, path):

        if not Path(path).exists():
            print("Checkpoint not found.")
            return None

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config = checkpoint["config"]

        print(f"Loaded epoch {checkpoint['epoch']}")

        return checkpoint



    def find_latest_checkpoint(self, checkpoint_dir="checkpoints"):
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            return None

        checkpoints = list(checkpoint_dir.glob("gpt_*.pt"))

        if not checkpoints:
            return None

        def get_epoch(path):
            match = re.search(r"gpt_(\d+)\.pt", path.name)

            if match is None:
                raise ValueError(f"Invalid checkpoint filename: {path.name}")

            return int(match.group(1))

        checkpoints.sort(key=get_epoch)

        return checkpoints[-1]

    def resume_training(self, epochs):

        latest = self.find_latest_checkpoint()

        if latest is None:
            print("No checkpoint found. Starting from scratch.")

            self.train(epochs)

            return

        print(f"Resuming from {latest}")

        checkpoint = self.load_checkpoint(latest)

        if checkpoint is None:
            return

        self.train(
            epochs=epochs,
            start_epoch=checkpoint["epoch"],
        )
