# test amp native

import torch
from torch.cuda.amp import autocast


def test():
    # Creates some tensors in default dtype (here assumed to be float32)
    a_float32 = torch.rand((8, 8), device="cuda")
    b_float32 = torch.rand((8, 8), device="cuda")
    c_float32 = torch.rand((8, 8), device="cuda")
    d_float32 = torch.rand((8, 8), device="cuda")

    with autocast():
        e_float16 = torch.mm(a_float32, b_float32)
        print(e_float16)
        with autocast(enabled=False):
            # Calls e_float16.float() to ensure float32 execution
            # (necessary because e_float16 was created in an autocasted region)
            f_float32 = torch.mm(c_float32, e_float16)

        # No manual casts are required when re-entering the autocast-enabled region.
        # torch.mm again runs in float16 and produces float16 output, regardless of input types.
        g_float16 = torch.mm(d_float32, f_float32)


if __name__ == '__main__':
    test()
