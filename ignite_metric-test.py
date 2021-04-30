import torch
import pytest
from ignite.engine import Engine
from ignite.metrics import Accuracy, VariableAccumulation, Average, GeometricAverage
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


# 1 1 28 28
# 64 10
def test():
    mean_var = GeometricAverage(device="cuda")
    criterion = nn.CrossEntropyLoss()
    m = nn.LogSoftmax(dim=1)
    model = Net()
    model = model.to("cuda")
    x = torch.randn((64, 1, 28, 28)).to("cuda")
    with autocast():
        output = model(x)
        output_ref = torch.randint(3, 5, (64,)).to("cuda")
        loss = criterion(output, output_ref)
        mean_var.update(loss)
        with autocast(enabled=False):
            # Calls e_float16.float() to ensure float32 execution
            # (necessary because e_float16 was created in an autocasted region)
            mean_var.update(torch.rand(1).double())
    # assert m.item() == pytest.approx(y_true.mean().item())


def test_apex():
    try:
        from apex import amp

        APEX_AVAILABLE = True
    except ModuleNotFoundError:
        APEX_AVAILABLE = False

    with amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
        scaled_loss.backward()


if __name__ == "__main__":
    test_apex()
