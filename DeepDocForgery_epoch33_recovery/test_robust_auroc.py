import unittest
import math
from robust_auroc import binary_auroc

class TestAUROC(unittest.TestCase):
    def test_perfect(self): self.assertEqual(binary_auroc([.1,.2,.8,.9],[0,0,1,1]),1.)
    def test_inverse(self): self.assertEqual(binary_auroc([.8,.9,.1,.2],[0,0,1,1]),0.)
    def test_ties(self): self.assertEqual(binary_auroc([.5,.5,.5,.5],[0,0,1,1]),.5)
    def test_nan(self): self.assertEqual(binary_auroc([float('nan'),.1,.9],[0,0,1]),1.)
    def test_inf(self): self.assertEqual(binary_auroc([float('inf'),.1,.9],[0,0,1]),1.)
    def test_undefined(self): self.assertTrue(math.isnan(binary_auroc([.2],[1])))

if __name__=='__main__': unittest.main()
