import torch
from ignite.engine import Engine
from ignite.engine import convert_tensor
from ignite.metrics import Average, GeometricAverage
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import wide_resnet50_2


# from utils import get_train_eval_loaders

class MockDataset(Dataset):
    def __init__(self, nb_samples=60, labels=100):
        self._labels = labels
        self._nb_samples = nb_samples

    def __len__(self):
        return self._nb_samples

    def __getitem__(self, index):
        x = torch.randn((3, 32, 32))
        y = torch.randint(0, 100, (1,)).item()
        return x, y


def get_train_eval_loaders(batch_size=8):
    """Setup the dataflow:
        - load CIFAR100 train and test datasets
        - setup train/test image transforms
            - horizontally flipped randomly and augmented using cutout.
            - each mini-batch contained 256 examples
        - setup train/test data loaders

    Returns:
        train_loader, test_loader, eval_train_loader
    """
    # train_transform = Compose(
    #     [
    #         Pad(4),
    #         RandomCrop(32),
    #         RandomHorizontalFlip(),
    #         ToTensor(),
    #         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         RandomErasing(),
    #     ]
    # )
    #
    # test_transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #
    # train_dataset = CIFAR100(root=path, train=True, transform=train_transform, download=True)
    # test_dataset = CIFAR100(root=path, train=False, transform=test_transform, download=False)
    #
    # train_eval_indices = [random.randint(0, len(train_dataset) - 1) for i in range(len(test_dataset))]
    # train_eval_dataset = Subset(train_dataset, train_eval_indices)
    #
    train_dataset = MockDataset()
    test_dataset = MockDataset()
    train_eval_dataset = MockDataset()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=12, shuffle=True, drop_last=True, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=12, shuffle=False, drop_last=False, pin_memory=True
    )

    eval_train_loader = DataLoader(
        train_eval_dataset, batch_size=batch_size, num_workers=12, shuffle=False, drop_last=False, pin_memory=True
    )

    return train_loader, test_loader, eval_train_loader


def main(batch_size=16, max_epochs=10):
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
    # torch.backends.cudnn.benchmark = True

    device = "cuda"

    train_loader, test_loader, eval_train_loader = get_train_eval_loaders(batch_size=batch_size)

    model = wide_resnet50_2(num_classes=100).to(device)
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss().to(device)

    scaler = GradScaler()

    def train_step(engine, batch):
        print(f"Epoch: {engine.state.iteration}")
        x = convert_tensor(batch[0], device, non_blocking=True)
        y = convert_tensor(batch[1], device, non_blocking=True)

        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(enabled=True):
            y_pred = model(x)
            # print("y_pred")
            # print(y_pred)
            print(y_pred.dtype)
            # print("y")
            # print(y.shape)
            # loss = criterion(y_pred, y)
            # print(loss.dtype)
        return y_pred

    trainer = Engine(train_step)
    metric = GeometricAverage(device='cuda')
    metric.attach(trainer, "Average")
    # ...
    #train_loader_= [(torch.randn((3, 32, 32)), torch.randint(0, 100, (1,))) for _ in range(16)]

    def infinite_iterator(batch_size, nb_samples=1):
        for sample in range(nb_samples):
            batch_x = torch.rand(batch_size, 3, 32, 32)
            batch_y = torch.randint( 0, 100, (batch_size,1,))
            yield batch_x, batch_y

    state = trainer.run(infinite_iterator(16,1))
    print(state.metrics['Average'])
    print(state.metrics['Average'].shape)
    print(state.metrics['Average'].dtype)

    metric.update(torch.randint( 0, 100, (1,)).to("cuda"))
    metric.compute()

if __name__ == "__main__":
    # fire.Fire(main)
    main()
