import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class QRNNLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, window_size=2, bidirectional=True):
        super().__init__()

        self.num_gates = 3
        self.window_size = window_size
        self.input_size = input_size
        self.output_size = output_size
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.fc = torch.nn.Linear(
                self.window_size * input_size, 2 * output_size * self.num_gates
            )
        else:
            self.fc = torch.nn.Linear(
                self.window_size * input_size, output_size * self.num_gates
            )

    def stride(self, x):
        """Create window_size sequences of embeddings.
        For the begining of sequence, insert padding embeddings to the left.
        For instance, sequence of window_size 3 will create the following output:

        Input: [1, 2, 3]
        Output: [[0, 0, 1]
                 [0, 1, 2]
                 [1, 2, 3]]
        """
        # Pad the left of bos
        pad_length = max(self.window_size - 1, 1)
        x = F.pad(x, (0, 0, pad_length, 0), value=0)

        bsz = x.size(0)
        seq_len = x.size(1)
        size = (
            bsz,
            seq_len - pad_length,
            self.input_size * self.window_size,
        )
        stride = x.stride()
        x = torch.as_strided(
            x,
            size,
            stride,
        )
        return x

    def forward(self, x):
        if isinstance(x, PackedSequence):
            return self.forward_packed_sequence(x)
        elif isinstance(x, torch.Tensor):
            return self.forward_tensor(x)

    def forward_packed_sequence(self, x):
        print(x)

    #         batch_sizes = x.batch_sizes
    #         num_steps = batch_sizes[0]
    #         print(batch_sizes)
    #         input_offset = 0
    #         pre_compute_input = self.fc(x.data)
    #         for i in range(num_steps):
    #             batch_size = batch_sizes[i]
    #             step_input = pre_compute_input.narrow(0, input_offset, batch_size)
    #             input_offset += batch_size
    #             print(i, batch_size, step_input.size())

    def forward_tensor(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (bsz, seq_len, input_size)
        """
        bsz = x.size(0)
        seq_len = x.size(1)

        x = self.stride(x)
        x = x.view(bsz, seq_len, -1)
        x = self.fc(x)
        z, f, o = x.chunk(self.num_gates, dim=2)

        z = torch.tanh(z)
        f = torch.sigmoid(f)
        seq_len = z.size(1)

        c = torch.zeros_like(z)

        if self.bidirectional:
            c = c.view(bsz, seq_len, 2, self.output_size)
            f = f.view(bsz, seq_len, 2, self.output_size)
            z = z.view(bsz, seq_len, 2, self.output_size)
            # Forward QRNN
            direction = 0
            for t in range(seq_len):
                f_t = f[:, t, direction]
                z_t = z[:, t, direction]
                if t == 0:
                    c[:, t, direction] = (1 - f_t) * z_t
                else:
                    c_prev = c[:, t - 1, direction].clone()
                    c[:, t, direction] = f_t * c_prev + (1 - f_t) * z_t
            # backward QRNN
            direction = 1
            for t in range(seq_len - 1, -1, -1):
                f_t = f[:, t, direction]
                z_t = z[:, t, direction]
                if t == seq_len - 1:
                    c[:, t, direction] = (1 - f_t) * z_t
                else:
                    c_prev = c[:, t + 1, direction].clone()
                    c[:, t, direction] = f_t * c_prev + (1 - f_t) * z_t
            c = c.view(bsz, seq_len, 2 * self.output_size)
        else:
            for t in range(seq_len):
                if t == 0:
                    c[:, t] = (1 - f[:, t]) * z[:, t]
                else:
                    c[:, t] = f[:, t] * c[:, t - 1].clone() + (1 - f[:, t]) * z[:, t]

        h = torch.sigmoid(o) * c
        return h
