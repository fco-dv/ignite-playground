import torch
from apex import amp
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import wide_resnet50_2
from ignite.engine import Engine, convert_tensor
from ignite.metrics import Average
from torch.utils.data import Dataset, DataLoader
from torch.nn import NLLLoss

class RndDataset(Dataset):
    def __init__(self, nb_samples=128, labels=100):
        self._labels = labels
        self._nb_samples = nb_samples

    def __len__(self):
        return self._nb_samples

    def __getitem__(self, index):
        x = torch.randn((3, 32, 32))
        y = torch.randint(0, 100, (1,)).item()
        return x, y

def main(opt="O1"):
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
    torch.backends.cudnn.benchmark = True

    device = "cuda"

    dataset = RndDataset()
    dataloader = DataLoader(dataset, batch_size=16)

    model = wide_resnet50_2(num_classes=100).to(device)
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = NLLLoss().to(device)

    model, optimizer = amp.initialize(model, optimizer, opt_level=opt)

    def train_step(engine, batch):
        x = convert_tensor(batch[0], device, non_blocking=True)
        y = convert_tensor(batch[1], device, non_blocking=True)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        return loss.item()


    trainer = Engine(train_step)
    metric = Average()
    metric.attach(trainer, "average")

    trainer.run(dataloader, max_epochs=1)


if __name__ == "__main__":
    main()