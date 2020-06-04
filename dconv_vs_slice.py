import sys
import time
import torch
from torch import nn
import torch.nn.functional as F

class DilatedConv(nn.Module):
    def __init__(self, n_elem, n_filt, n_dil, n_in_chan, n_out_chan):
        super(DilatedConv, self).__init__()
        self.register_buffer('weight', torch.randn(n_out_chan, n_in_chan,
            n_filt))
        nn.init.xavier_uniform_(self.weight)
        self.n_dil = n_dil

    def forward(self, x):
        """
        Incrementally loop through, starting from the beginning of x,
        filling in each new value from 'a'
        x: n_batch, n_in_chan, n_elem 
        
        """
        n_filt = self.weight.shape[2]
        n_out_chan = self.weight.shape[0]

        filter_size = (n_filt - 1) * self.n_dil + 1
        a = torch.randn(x.shape[0], n_out_chan, x.shape[2] - filter_size,
                device=x.device)
        n_steps = a.shape[2]

        for i in range(n_steps):
            a[...,i] = F.conv1d(x[...,i:i+filter_size], self.weight, bias=None,
                    stride=1, padding=0, dilation=self.n_dil).squeeze(2)
            # b = F.conv1d(x[...,i:i+filter_size:self.n_dil], self.weight, bias=None,
            #         stride=1, padding=0, dilation=1).squeeze(2)
            # if not torch.equal(a[...,i], b):
            #     print('a: ', a[...,i])
            #     print('b: ', b)
            #     print('on step {} out of {}'.format(i, n_steps))
            #     return

            x[...,i+1] = a[...,i]
        return x


class StridedConv(nn.Module):
    def __init__(self, n_elem, n_filt, n_dil, n_in_chan, n_out_chan):
        super(StridedConv, self).__init__()
        self.register_buffer('weight', torch.randn(n_out_chan, n_in_chan,
            n_filt))
        nn.init.xavier_uniform_(self.weight)
        self.n_dil = n_dil

    def forward(self, x):
        """
        Incrementally loop through, starting from the beginning of x,
        filling in each new value from 'a'
        x: n_batch, n_in_chan, n_elem 
        
        """
        n_filt = self.weight.shape[2]
        n_out_chan = self.weight.shape[0]

        filter_size = (n_filt - 1) * self.n_dil + 1
        a = torch.randn(x.shape[0], n_out_chan, x.shape[2] - filter_size,
                device=x.device)
        n_steps = a.shape[2]

        for i in range(n_steps):
            a[...,i] = F.conv1d(x[...,i:i+filter_size:self.n_dil], self.weight, bias=None,
                    stride=1, padding=0, dilation=1).squeeze(2)
            x[...,i+1] = a[...,i]
        return x



def main():
    n_elem = int(sys.argv[1])
    n_dil = int(sys.argv[2])
    n_batch = 10
    n_filt = 2 
    n_in_chan = 256
    n_out_chan = 256 
    dc = DilatedConv(n_elem, n_filt, n_dil, n_in_chan, n_out_chan)
    sc = StridedConv(n_elem, n_filt, n_dil, n_in_chan, n_out_chan)
    dc_scr = torch.jit.script(dc)
    sc_scr = torch.jit.script(sc)

    source = torch.randn(n_batch, n_in_chan, n_elem)
    x = source

    dev = torch.device('cuda')
    dc.to(dev)
    sc.to(dev)
    dc_scr.to(dev)
    sc_scr.to(dev)

    x = x.to(dev)

    with torch.no_grad():
        start_time = time.perf_counter()
        y = dc(x)
        print(f"finished Dilated Conv in {time.perf_counter() - start_time:0.4f} seconds")

        start_time = time.perf_counter()
        z = sc(x)
        print(f"finished Strided Conv in {time.perf_counter() - start_time:0.4f} seconds")

        start_time = time.perf_counter()
        z = dc_scr(x)
        print(f"finished Dilated Conv Script in {time.perf_counter() - start_time:0.4f} seconds")

        start_time = time.perf_counter()
        z = sc_scr(x)
        print(f"finished Strided Conv Script in {time.perf_counter() - start_time:0.4f} seconds")

if __name__ == '__main__':
    main()

