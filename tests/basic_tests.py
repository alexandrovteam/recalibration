import unittest
import numpy as np

class UtilTests(unittest.TestCase):
    def test_find_nearest_single(self):
        from recalibration.utils import find_nearest
        np.testing.assert_equal(find_nearest(1.,1.), (0,1))

    def test_find_nearest_one_vs_vector(self):
        from recalibration.utils import find_nearest
        x1 = np.linspace(0,10,100)
        x2 = [0, 1, 5.5, 7, 11]
        for _x in x2:
            with self.subTest(i=_x):
                true_index = np.argmin(np.abs(x1-_x))
                true_value = x1[true_index]
                self.assertEqual(find_nearest(x1, _x), (true_index, true_value))

    def test_find_nearest_vector(self):
        from recalibration.utils import find_nearest
        x1 = np.linspace(0, 10, 100)
        x2 = [0, 1, 5.5, 7]
        true_indexs = []
        for _x in x2:
            true_indexs.append( np.argmin(np.abs(x1 - _x)) )
        true_indexs = np.asarray(true_indexs)
        np.testing.assert_equal(find_nearest(x1, x2)[0], true_indexs)
        np.testing.assert_equal(find_nearest(x1, x2)[1], x1[true_indexs])


    def test_get_deltas(self):
        from recalibration.utils import get_deltas
        inputs = [(300, 300, 0),
                    (300, 300.0003, 1),
                    ]
        for args in inputs:
            with self.subTest():
                self.assertAlmostEqual(get_deltas(args[0], args[1],), args[2], 5)


    def test_get_deltas_mix(self):
        from recalibration.utils import get_deltas_mix
        inputs = [([300.], [300.], 0),
                  ([300.0003], [300.], -1),
                  ([300.0006], [300.], -2),
                  ([300.0015], [300.], -5),
                  ([300.003], [300.], -10),
                  ([299.9997], [300.], 1),
                  ([299.9994], [300.], 2),
                  ([299.9985], [300.], 5),
                  ([299.997], [300.], 10),
                  ]
        for args in inputs:
            with self.subTest():
                np.testing.assert_array_almost_equal(get_deltas_mix(args[0], args[1]), args[2])

    def test_get_deltas_mix_shape(self):
        from recalibration.utils import get_deltas_mix
        inputs = [
            (1, 1),
            (1, 10),
            (200, 100)
                  ]
        for args in inputs:
            with self.subTest():
                t0 = np.linspace(100,1000, args[0])
                t1 = np.linspace(100, 1000, args[1])
                self.assertEquals(len(get_deltas_mix(t0, t1)), args[1])

