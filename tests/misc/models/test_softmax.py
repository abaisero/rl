import unittest

import indextools
from rl.misc.models import Softmax


class SoftmaxTest(unittest.TestCase):
    def setUp(self):
        self.s1 = indextools.BoolSpace()
        self.s2 = indextools.DomainSpace('abc')
        self.s3 = indextools.RangeSpace(4)
        self.s4 = indextools.DomainSpace('defgh')

        self.model = Softmax(self.s1, self.s2, cond=(self.s3, self.s4))

    def test_init(self):
        s1, s2, s3, s4 = self.s1, self.s2, self.s3, self.s4
        model = self.model

        self.assertEqual(model.xdims, (4, 5))
        self.assertEqual(model.ydims, (2, 3))
        self.assertEqual(model.dims, (4, 5, 2, 3))
        self.assertEqual(model.xsize, 20)
        self.assertEqual(model.ysize, 6)
        self.assertEqual(model.size, 120)
        self.assertEqual(model.xaxes, (0, 1))
        self.assertEqual(model.yaxes, (2, 3))

    def test_params(self):
        s1, s2, s3, s4 = self.s1, self.s2, self.s3, self.s4
        model = self.model

        self.assertEqual(model.params.shape, (4, 5, 2, 3))

    # TODO test gradients!!!

#     def test_prefs(self):
#         s1, s2, s3, s4 = self.s1, self.s2, self.s3, self.s4
#         model = self.model

        # assertEquaj

    def test_prefs_shape(self):
        model = self.model
        e1 = self.s1.elem(0)
        e2 = self.s2.elem(0)
        e3 = self.s3.elem(0)
        e4 = self.s4.elem(0)

        __ = slice(None)
        self.assertEqual(model.prefs(__, __, __, __).shape, (4, 5, 2, 3))
        self.assertEqual(model.prefs(__, __, __, e4).shape, (4, 5, 2   ))
        self.assertEqual(model.prefs(__, __, e3, __).shape, (4, 5,    3))
        self.assertEqual(model.prefs(__, __, e3, e4).shape, (4, 5      ))
        self.assertEqual(model.prefs(__, e2, __, __).shape, (4,    2, 3))
        self.assertEqual(model.prefs(__, e2, __, e4).shape, (4,    2   ))
        self.assertEqual(model.prefs(__, e2, e3, __).shape, (4,       3))
        self.assertEqual(model.prefs(__, e2, e3, e4).shape, (4,        ))
        self.assertEqual(model.prefs(e1, __, __, __).shape, (   5, 2, 3))
        self.assertEqual(model.prefs(e1, __, __, e4).shape, (   5, 2   ))
        self.assertEqual(model.prefs(e1, __, e3, __).shape, (   5,    3))
        self.assertEqual(model.prefs(e1, __, e3, e4).shape, (   5,     ))
        self.assertEqual(model.prefs(e1, e2, __, __).shape, (      2, 3))
        self.assertEqual(model.prefs(e1, e2, __, e4).shape, (      2,  ))
        self.assertEqual(model.prefs(e1, e2, e3, __).shape, (         3,))
        self.assertEqual(model.prefs(e1, e2, e3, e4).shape, (          ))


    def test_logprobs_shape(self):
        model = self.model
        e1 = self.s1.elem(0)
        e2 = self.s2.elem(0)
        e3 = self.s3.elem(0)
        e4 = self.s4.elem(0)

        __ = slice(None)
        self.assertEqual(model.logprobs(__, __, __, __).shape, (4, 5, 2, 3))
        self.assertEqual(model.logprobs(__, __, __, e4).shape, (4, 5, 2   ))
        self.assertEqual(model.logprobs(__, __, e3, __).shape, (4, 5,    3))
        self.assertEqual(model.logprobs(__, __, e3, e4).shape, (4, 5      ))
        self.assertEqual(model.logprobs(__, e2, __, __).shape, (4,    2, 3))
        self.assertEqual(model.logprobs(__, e2, __, e4).shape, (4,    2   ))
        self.assertEqual(model.logprobs(__, e2, e3, __).shape, (4,       3))
        self.assertEqual(model.logprobs(__, e2, e3, e4).shape, (4,        ))
        self.assertEqual(model.logprobs(e1, __, __, __).shape, (   5, 2, 3))
        self.assertEqual(model.logprobs(e1, __, __, e4).shape, (   5, 2   ))
        self.assertEqual(model.logprobs(e1, __, e3, __).shape, (   5,    3))
        self.assertEqual(model.logprobs(e1, __, e3, e4).shape, (   5,     ))
        self.assertEqual(model.logprobs(e1, e2, __, __).shape, (      2, 3))
        self.assertEqual(model.logprobs(e1, e2, __, e4).shape, (      2,  ))
        self.assertEqual(model.logprobs(e1, e2, e3, __).shape, (         3,))
        self.assertEqual(model.logprobs(e1, e2, e3, e4).shape, (          ))

    def test_probs_shape(self):
        model = self.model
        e1 = self.s1.elem(0)
        e2 = self.s2.elem(0)
        e3 = self.s3.elem(0)
        e4 = self.s4.elem(0)

        __ = slice(None)
        self.assertEqual(model.probs(__, __, __, __).shape, (4, 5, 2, 3))
        self.assertEqual(model.probs(__, __, __, e4).shape, (4, 5, 2   ))
        self.assertEqual(model.probs(__, __, e3, __).shape, (4, 5,    3))
        self.assertEqual(model.probs(__, __, e3, e4).shape, (4, 5      ))
        self.assertEqual(model.probs(__, e2, __, __).shape, (4,    2, 3))
        self.assertEqual(model.probs(__, e2, __, e4).shape, (4,    2   ))
        self.assertEqual(model.probs(__, e2, e3, __).shape, (4,       3))
        self.assertEqual(model.probs(__, e2, e3, e4).shape, (4,        ))
        self.assertEqual(model.probs(e1, __, __, __).shape, (   5, 2, 3))
        self.assertEqual(model.probs(e1, __, __, e4).shape, (   5, 2   ))
        self.assertEqual(model.probs(e1, __, e3, __).shape, (   5,    3))
        self.assertEqual(model.probs(e1, __, e3, e4).shape, (   5,     ))
        self.assertEqual(model.probs(e1, e2, __, __).shape, (      2, 3))
        self.assertEqual(model.probs(e1, e2, __, e4).shape, (      2,  ))
        self.assertEqual(model.probs(e1, e2, e3, __).shape, (         3,))
        self.assertEqual(model.probs(e1, e2, e3, e4).shape, (          ))

    # @unittest.skip
    def test_dprefs_shape(self):
        model = self.model
        e1 = self.s1.elem(0)
        e2 = self.s2.elem(0)
        e3 = self.s3.elem(0)
        e4 = self.s4.elem(0)

        __ = slice(None)
        self.assertEqual(model.dprefs(__, __, __, __).shape, (4, 5, 2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dprefs(__, __, __, e4).shape, (4, 5, 2,    4, 5, 2, 3))
        self.assertEqual(model.dprefs(__, __, e3, __).shape, (4, 5,    3, 4, 5, 2, 3))
        self.assertEqual(model.dprefs(__, __, e3, e4).shape, (4, 5,       4, 5, 2, 3))
        self.assertEqual(model.dprefs(__, e2, __, __).shape, (4,    2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dprefs(__, e2, __, e4).shape, (4,    2,    4, 5, 2, 3))
        self.assertEqual(model.dprefs(__, e2, e3, __).shape, (4,       3, 4, 5, 2, 3))
        self.assertEqual(model.dprefs(__, e2, e3, e4).shape, (4,          4, 5, 2, 3))
        self.assertEqual(model.dprefs(e1, __, __, __).shape, (   5, 2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dprefs(e1, __, __, e4).shape, (   5, 2,    4, 5, 2, 3))
        self.assertEqual(model.dprefs(e1, __, e3, __).shape, (   5,    3, 4, 5, 2, 3))
        self.assertEqual(model.dprefs(e1, __, e3, e4).shape, (   5,       4, 5, 2, 3))
        self.assertEqual(model.dprefs(e1, e2, __, __).shape, (      2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dprefs(e1, e2, __, e4).shape, (      2,    4, 5, 2, 3))
        self.assertEqual(model.dprefs(e1, e2, e3, __).shape, (         3, 4, 5, 2, 3))
        self.assertEqual(model.dprefs(e1, e2, e3, e4).shape, (            4, 5, 2, 3))

    # @unittest.skip
    def test_dlogprobs_shape(self):
        model = self.model
        e1 = self.s1.elem(0)
        e2 = self.s2.elem(0)
        e3 = self.s3.elem(0)
        e4 = self.s4.elem(0)

        __ = slice(None)
        self.assertEqual(model.dlogprobs(__, __, __, __).shape, (4, 5, 2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(__, __, __, e4).shape, (4, 5, 2,    4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(__, __, e3, __).shape, (4, 5,    3, 4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(__, __, e3, e4).shape, (4, 5,       4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(__, e2, __, __).shape, (4,    2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(__, e2, __, e4).shape, (4,    2,    4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(__, e2, e3, __).shape, (4,       3, 4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(__, e2, e3, e4).shape, (4,          4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(e1, __, __, __).shape, (   5, 2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(e1, __, __, e4).shape, (   5, 2,    4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(e1, __, e3, __).shape, (   5,    3, 4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(e1, __, e3, e4).shape, (   5,       4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(e1, e2, __, __).shape, (      2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(e1, e2, __, e4).shape, (      2,    4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(e1, e2, e3, __).shape, (         3, 4, 5, 2, 3))
        self.assertEqual(model.dlogprobs(e1, e2, e3, e4).shape, (            4, 5, 2, 3))

    # @unittest.skip
    def test_dprobs_shape(self):
        model = self.model
        e1 = self.s1.elem(0)
        e2 = self.s2.elem(0)
        e3 = self.s3.elem(0)
        e4 = self.s4.elem(0)

        __ = slice(None)
        self.assertEqual(model.dprobs(__, __, __, __).shape, (4, 5, 2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dprobs(__, __, __, e4).shape, (4, 5, 2,    4, 5, 2, 3))
        self.assertEqual(model.dprobs(__, __, e3, __).shape, (4, 5,    3, 4, 5, 2, 3))
        self.assertEqual(model.dprobs(__, __, e3, e4).shape, (4, 5,       4, 5, 2, 3))
        self.assertEqual(model.dprobs(__, e2, __, __).shape, (4,    2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dprobs(__, e2, __, e4).shape, (4,    2,    4, 5, 2, 3))
        self.assertEqual(model.dprobs(__, e2, e3, __).shape, (4,       3, 4, 5, 2, 3))
        self.assertEqual(model.dprobs(__, e2, e3, e4).shape, (4,          4, 5, 2, 3))
        self.assertEqual(model.dprobs(e1, __, __, __).shape, (   5, 2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dprobs(e1, __, __, e4).shape, (   5, 2,    4, 5, 2, 3))
        self.assertEqual(model.dprobs(e1, __, e3, __).shape, (   5,    3, 4, 5, 2, 3))
        self.assertEqual(model.dprobs(e1, __, e3, e4).shape, (   5,       4, 5, 2, 3))
        self.assertEqual(model.dprobs(e1, e2, __, __).shape, (      2, 3, 4, 5, 2, 3))
        self.assertEqual(model.dprobs(e1, e2, __, e4).shape, (      2,    4, 5, 2, 3))
        self.assertEqual(model.dprobs(e1, e2, e3, __).shape, (         3, 4, 5, 2, 3))
        self.assertEqual(model.dprobs(e1, e2, e3, e4).shape, (            4, 5, 2, 3))
