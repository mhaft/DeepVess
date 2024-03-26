import unittest

import topologylayer
import torch
import numpy as np
from topologylayer.util.process import remove_zero_bars, remove_infinite_bars

class RipsTest(unittest.TestCase):
    def test(self):
        from topologylayer.nn import RipsLayer

        for alg in ['hom', 'hom2', 'cohom']:
            # superlevel set
            layer = RipsLayer(4, maxdim=1, alg=alg)
            x = torch.tensor([[1, 1], [1,-1], [-1,-1], [-1,1]], dtype=torch.float).requires_grad_(True)

            dgms, issub = layer(x)
            self.assertEqual(
                issub,
                True,
                "Expected sublevel set layer. alg=" + alg)
            self.assertEqual(
                torch.all(torch.eq(remove_infinite_bars(remove_zero_bars(dgms[0]), issub),
                            torch.tensor([[0., 2.], [0., 2.], [0., 2.]]))),
                True,
                "unexpected 0-dim barcode. alg=" + alg)
            self.assertEqual(
                torch.all(torch.eq(remove_zero_bars(dgms[1]),
                            torch.tensor([[2., 2.8284270763397217]]))),
                True,
                "unexpected 1-dim barcode. alg=" + alg)

            d0 = remove_infinite_bars(remove_zero_bars(dgms[0]), issub)
            p = torch.sum(d0[:, 1] - d0[:, 0])
            p.backward()

            self.assertEqual(
                torch.all(torch.eq(x.grad,
                    torch.tensor([[1,1],[1,-1],[-1,0],[-1,0]], dtype=torch.float))),
                True,
                "unexpected gradient. alg=" + alg)
