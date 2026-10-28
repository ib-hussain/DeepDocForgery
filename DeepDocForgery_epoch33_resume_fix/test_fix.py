import math
import unittest
import warnings
from deepdocforgery.metrics import binary_auroc

class TestAUROC(unittest.TestCase):
    def test_perfect(self): self.assertEqual(binary_auroc([0.1,0.2,0.8,0.9],[0,0,1,1]),1.0)
    def test_inverse(self): self.assertEqual(binary_auroc([0.8,0.9,0.1,0.2],[0,0,1,1]),0.0)
    def test_ties(self): self.assertEqual(binary_auroc([0.5,0.5,0.5,0.5],[0,0,1,1]),0.5)
    def test_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.assertEqual(binary_auroc([float('nan'),0.1,0.9],[0,0,1]),1.0)
    def test_inf(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.assertEqual(binary_auroc([float('inf'),0.1,0.9],[0,0,1]),1.0)
    def test_empty_class(self): self.assertIsNone(binary_auroc([0.1],[0]))
    def test_empty(self): self.assertIsNone(binary_auroc([],[]))
    def test_length_error(self):
        with self.assertRaises(ValueError): binary_auroc([0.2],[0,1])

if __name__ == '__main__': unittest.main()
