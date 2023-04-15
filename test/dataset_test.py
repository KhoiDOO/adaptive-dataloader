import unittest

class MultiplicationTestCase(unittest.TestCase):

    def setUp(self):
        from adaptive_dataset.dataset.mnist import Mnist
        from adaptive_dataset.augment.aug import LargeAugmentation
        from adaptive_dataset.augment.aug import SegAumentation
        from adaptive_dataset import LargeAugmentation
        from adaptive_dataset import SegAumentation
        from adaptive_dataset import DlakeInterface

if __name__ == '__main__':
    unittest.main()