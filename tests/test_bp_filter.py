import unittest
from dasQt.filter.bpFilter import bp_filter
import numpy as np

# def bp_filter(d, dt, f1, f2, f3, f4):


class TestBpFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.d = np.sin(np.linspace(0, 10, 1000))
        print(self.d.shape)
        self.dt = 0.01
        self.f1 = 0.1
        self.f2 = 0.2
        self.f3 = 10.0
        self.f4 = 11.0
        
        return super().setUp()
    
    def test_bp_filter(self):
        # self.asserEqual is a method that checks if the two arguments are equal
        self.assertEqual(bp_filter(self.d, self.dt, self.f1, self.f2, self.f3, self.f4).shape, 
                         self.d.shape)
        
    def tearDown(self) -> None:
        del self.d, self.dt, self.f1, self.f2, self.f3, self.f4
        return super().tearDown()
    
if __name__ == '__main__':
    unittest.main()