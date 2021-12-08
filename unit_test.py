import unittest
from unittest import mock
import os
import numpy as np

import bifrost

unittest.TestLoader.sortTestMethodsUsing = None


class TestSpectrum(unittest.TestCase):

    spectrum1 = bifrost.Spectrum.from_fits(os.path.join(os.path.dirname(__file__), 'examples', 'example_data',
                                                       'spec-1171-52753-0076.fits'), 'Unittest-Spectrum-1')
    spectrum2 = bifrost.Spectrum.simulated(5000, (-50, 50), rv='random', vsini='random', noise_std='random',
                                          a=5, sigma=0.001, name='Unittest-Spectrum-2')
    spectrum3 = bifrost.Spectrum.simulated(4950, (-50, 50), rv='random', vsini='random', noise_std='random',
                                           a=3, sigma=0.5, name='Unittest-Spectrum-3')

    spectra = bifrost.Spectra()
    spectra.add_spec(spectrum1)
    spectra.add_spec(spectrum2)

    stack = bifrost.Stack()
    stack.add_spec(spectrum2)
    stack.add_spec(spectrum3)
    stack.correct_spectra()

    def test_fromfits_constructor(self):
        self.assertIsInstance(TestSpectrum.spectrum1.wave, np.ndarray)
        self.assertIsInstance(TestSpectrum.spectrum1.flux, np.ndarray)
        self.assertIsInstance(TestSpectrum.spectrum1.error, np.ndarray)
        self.assertTrue((TestSpectrum.spectrum1.wave.size == TestSpectrum.spectrum1.flux.size)
                        and (TestSpectrum.spectrum1.flux.size == TestSpectrum.spectrum1.error.size))
        self.assertEqual(TestSpectrum.spectrum1.name, 'Unittest-Spectrum-1')
        self.assertAlmostEqual(TestSpectrum.spectrum1.redshift, 0.033102997)
        self.assertAlmostEqual(TestSpectrum.spectrum1.ra, 16.304674666666667)
        self.assertAlmostEqual(TestSpectrum.spectrum1.dec, 41.293068)
        self.assertAlmostEqual(TestSpectrum.spectrum1.ebv, 0.0071)
        self.assertIsInstance(TestSpectrum.spectrum1.data, dict)

    def test_simulated_constructor(self):
        self.assertIsInstance(TestSpectrum.spectrum2.wave, np.ndarray)
        self.assertIsInstance(TestSpectrum.spectrum2.flux, np.ndarray)
        self.assertIsInstance(TestSpectrum.spectrum2.error, np.ndarray)
        self.assertTrue((TestSpectrum.spectrum2.wave.size == TestSpectrum.spectrum2.flux.size)
                        and (TestSpectrum.spectrum2.flux.size == TestSpectrum.spectrum2.error.size))
        self.assertEqual(TestSpectrum.spectrum2.name, 'Unittest-Spectrum-2')
        self.assertIsNone(TestSpectrum.spectrum2.ra)
        self.assertIsNone(TestSpectrum.spectrum2.dec)
        self.assertIsNone(TestSpectrum.spectrum2.ebv)
        self.assertIsInstance(TestSpectrum.spectrum2.redshift, float)
        self.assertIsInstance(TestSpectrum.spectrum2.data, dict)

    def test_corrections(self):
        TestSpectrum.spectrum1.apply_corrections()
        self.assertTrue(TestSpectrum.spectrum1.corrected)

    def test_snr(self):
        TestSpectrum.spectrum1.calc_snr()
        self.assertTrue('snr' in TestSpectrum.spectrum1.data)

    def test_line_snr(self):
        TestSpectrum.spectrum1.calc_line_snr(wave_range=(5000, 5010), key='snr_5005')
        self.assertTrue('snr_5005' in TestSpectrum.spectrum1.data)

    def test_calc_agn_frac(self):
        TestSpectrum.spectrum1.data['a'] = 0.1
        TestSpectrum.spectrum1.data['b'] = 0.1
        TestSpectrum.spectrum1.calc_agn_frac('a', 'b', (0.5, 0.5))
        self.assertAlmostEqual(TestSpectrum.spectrum1.data["agn_frac"], 1.7677669529663684)
        self.assertTrue(TestSpectrum.spectrum1.k01_agn_class('a', 'b'))

    # @mock.patch("%s.bifrost.spectrum.plt" % __name__)
    # def test_plot(self):
    #     TestSpectrum.spectrum1.plot(fname=os.path.join(__file__, 'test'), backend='pyplot')
    #     mock_plt.title.assert_called_once_with()

    def test_to_numpy(self):
        out = TestSpectrum.spectrum1.to_numpy(_slice=slice(0, 10))
        np.testing.assert_array_equal(out['wave'], TestSpectrum.spectrum1.wave[0:10])
        np.testing.assert_array_equal(out['flux'], TestSpectrum.spectrum1.flux[0:10])
        np.testing.assert_array_equal(out['err'], TestSpectrum.spectrum1.error[0:10])

    def test_add(self):
        TestSpectrum.spectrum1.apply_corrections()
        result = TestSpectrum.spectrum1 + TestSpectrum.spectrum1
        np.testing.assert_array_equal(result.wave, TestSpectrum.spectrum1.wave)
        np.testing.assert_array_almost_equal(result.flux, TestSpectrum.spectrum1.flux*2)
        np.testing.assert_array_almost_equal(result.error, np.hypot(TestSpectrum.spectrum1.error, TestSpectrum.spectrum1.error))

    def test_sub(self):
        TestSpectrum.spectrum1.apply_corrections()
        result = TestSpectrum.spectrum1 - TestSpectrum.spectrum1
        np.testing.assert_array_equal(result.wave, TestSpectrum.spectrum1.wave)
        np.testing.assert_array_almost_equal(result.flux, np.zeros(result.flux.size))
        np.testing.assert_array_almost_equal(result.error, np.hypot(TestSpectrum.spectrum1.error, TestSpectrum.spectrum1.error))

    def test_mult(self):
        TestSpectrum.spectrum1.apply_corrections()
        result = TestSpectrum.spectrum1 * TestSpectrum.spectrum1
        np.testing.assert_array_equal(result.wave, TestSpectrum.spectrum1.wave)
        np.testing.assert_array_almost_equal(result.flux, TestSpectrum.spectrum1.flux**2)
        np.testing.assert_array_almost_equal(result.error, np.hypot(TestSpectrum.spectrum1.error*TestSpectrum.spectrum1.flux,
                                                                    TestSpectrum.spectrum1.error*TestSpectrum.spectrum1.flux))

    def test_truediv(self):
        TestSpectrum.spectrum1.apply_corrections()
        result = TestSpectrum.spectrum1 / TestSpectrum.spectrum1
        np.testing.assert_array_equal(result.wave, TestSpectrum.spectrum1.wave)
        np.testing.assert_array_almost_equal(result.flux, np.ones(result.flux.size))
        np.testing.assert_array_almost_equal(result.error, (TestSpectrum.spectrum1.flux/TestSpectrum.spectrum1.flux)*
                                             np.hypot(TestSpectrum.spectrum1.error/TestSpectrum.spectrum1.flux,
                                             TestSpectrum.spectrum1.error/TestSpectrum.spectrum1.flux))

    def test_spectra_to_numpy(self):
        np_out = TestSpectrum.spectra.to_numpy(['wave', 'flux', 'error'])
        np.testing.assert_array_equal(np_out['wave'][0], TestSpectrum.spectrum1.wave)
        np.testing.assert_array_equal(np_out['wave'][1], TestSpectrum.spectrum2.wave)
        np.testing.assert_array_equal(np_out['flux'][0], TestSpectrum.spectrum1.flux)
        np.testing.assert_array_equal(np_out['flux'][1], TestSpectrum.spectrum2.flux)
        np.testing.assert_array_equal(np_out['error'][0], TestSpectrum.spectrum1.error)
        np.testing.assert_array_equal(np_out['error'][1], TestSpectrum.spectrum2.error)

    def test_add_del_spec(self):
        test_spec = bifrost.Spectrum(wave=np.linspace(0, 100, 101), flux=np.ones(101), error=np.ones(101)*0.1, name='c')
        orig = len(TestSpectrum.spectra)
        TestSpectrum.spectra.add_spec(test_spec)
        self.assertEqual(len(TestSpectrum.spectra), orig+1)
        self.assertIsInstance(TestSpectrum.spectra['c'], bifrost.Spectrum)
        del TestSpectrum.spectra['c']
        self.assertEqual(len(TestSpectrum.spectra), orig)

    def test_correct_spec(self):
        TestSpectrum.spectra.correct_spectra()
        for spec in TestSpectrum.spectra.values():
            self.assertTrue(spec.corrected)
        self.assertTrue(TestSpectrum.spectra.corrected)

    def test_spec_index(self):
        ind = TestSpectrum.spectra.get_spec_index('Unittest-Spectrum-1')
        self.assertEqual(ind, 0)
        name = TestSpectrum.spectra.get_spec_name(0)
        self.assertEqual(name, 'Unittest-Spectrum-1')

    def test_zzz_stack_1(self):
        TestSpectrum.stack.wave_criterion = 'lenient'
        TestSpectrum.stack()
        self.assertEqual(len(TestSpectrum.stack.universal_grid), 1)
        self.assertEqual(len(TestSpectrum.stack.stacked_flux), 1)
        self.assertEqual(len(TestSpectrum.stack.stacked_err), 1)
        self.assertIsNone(TestSpectrum.stack.binned)
        self.assertIsNone(TestSpectrum.stack.binned_spec)
        self.assertIsNone(TestSpectrum.stack.bin_counts)
        self.assertIsNone(TestSpectrum.stack.bin_edges)
        self.assertEqual(len(TestSpectrum.stack.universal_grid[0]), len(TestSpectrum.stack.stacked_flux[0]))
        self.assertEqual(len(TestSpectrum.stack.stacked_flux[0]), len(TestSpectrum.stack.stacked_err[0]))
        self.assertEqual(len(TestSpectrum.stack.specnames_f[0]), len(TestSpectrum.stack.stacked_flux[0]))
        self.assertEqual(len(TestSpectrum.stack.specnames_e[0]), len(TestSpectrum.stack.specnames_f[0]))
        self.assertTrue(all([n._normalized for n in TestSpectrum.stack.values()]))

    def test_zzz_stack_2(self):
        TestSpectrum.stack.wave_criterion = 'strict'
        TestSpectrum.stack()
        self.assertEqual(len(TestSpectrum.stack.universal_grid), 1)
        self.assertEqual(len(TestSpectrum.stack.stacked_flux), 1)
        self.assertEqual(len(TestSpectrum.stack.stacked_err), 1)
        self.assertIsNone(TestSpectrum.stack.binned)
        self.assertIsNone(TestSpectrum.stack.binned_spec)
        self.assertIsNone(TestSpectrum.stack.bin_counts)
        self.assertIsNone(TestSpectrum.stack.bin_edges)
        self.assertEqual(len(TestSpectrum.stack.universal_grid[0]), len(TestSpectrum.stack.stacked_flux[0]))
        self.assertEqual(len(TestSpectrum.stack.stacked_flux[0]), len(TestSpectrum.stack.stacked_err[0]))
        self.assertEqual(len(TestSpectrum.stack.specnames_f[0]), len(TestSpectrum.stack.stacked_flux[0]))
        self.assertEqual(len(TestSpectrum.stack.specnames_e[0]), len(TestSpectrum.stack.specnames_f[0]))
        self.assertEqual(len(TestSpectrum.stack.nspec_f[0]), len(TestSpectrum.stack.nspec_e[0]))
        self.assertTrue(all([n._normalized for n in TestSpectrum.stack.values()]))

    def test_zzz_stack_3(self):
        TestSpectrum.stack.wave_criterion = 'lenient'
        TestSpectrum.stack[0].data['a'] = 0.1
        TestSpectrum.stack[0].data['b'] = 0.2
        TestSpectrum.stack[1].data['a'] = 0.5
        TestSpectrum.stack[1].data['b'] = 0.6
        TestSpectrum.stack(bin_name='agn_frac', bpt_1='a', bpt_2='b', nbins=3, log=False, hbin_target=1)
        self.assertEqual(len(TestSpectrum.stack.universal_grid), 3)
        self.assertEqual(len(TestSpectrum.stack.stacked_flux), 3)
        self.assertEqual(len(TestSpectrum.stack.stacked_err), 3)
        self.assertEqual(TestSpectrum.stack.binned, 'agn_frac')
        self.assertEqual(len(TestSpectrum.stack.binned_spec), 3)
        self.assertEqual(len(TestSpectrum.stack.bin_counts), 3)
        self.assertEqual(len(TestSpectrum.stack.bin_edges), 4)
        self.assertEqual(len(TestSpectrum.stack.universal_grid[0]), len(TestSpectrum.stack.stacked_flux[0]))
        self.assertEqual(len(TestSpectrum.stack.stacked_flux[0]), len(TestSpectrum.stack.stacked_err[0]))
        self.assertEqual(len(TestSpectrum.stack.specnames_e[0]), len(TestSpectrum.stack.specnames_f[0]))
        self.assertEqual(len(TestSpectrum.stack.nspec_f[0]), len(TestSpectrum.stack.nspec_e[0]))
        self.assertTrue(all([n._normalized for n in TestSpectrum.stack.values()]))

    def test_zzz_stack_4(self):
        TestSpectrum.stack.wave_criterion = 'lenient'
        TestSpectrum.stack[0].data['a'] = 0.1
        TestSpectrum.stack[0].data['b'] = 0.2
        TestSpectrum.stack[1].data['a'] = 0.5
        TestSpectrum.stack[1].data['b'] = 0.6
        TestSpectrum.stack(bpt_1='a', bpt_2='b', stack_all_agns=True)
        self.assertEqual(len(TestSpectrum.stack.universal_grid), 1)
        self.assertEqual(len(TestSpectrum.stack.stacked_flux), 1)
        self.assertEqual(len(TestSpectrum.stack.stacked_err), 1)
        self.assertEqual(len(TestSpectrum.stack.universal_grid[0]), len(TestSpectrum.stack.stacked_flux[0]))
        self.assertEqual(len(TestSpectrum.stack.stacked_flux[0]), len(TestSpectrum.stack.stacked_err[0]))
        self.assertEqual(len(TestSpectrum.stack.specnames_e[0]), len(TestSpectrum.stack.specnames_f[0]))
        self.assertEqual(len(TestSpectrum.stack.nspec_f[0]), len(TestSpectrum.stack.nspec_e[0]))
        self.assertTrue(all([n._normalized for n in TestSpectrum.stack.values()]))

    def test_zzy_line_flux_ratios(self):
        _wl, _wr, out = TestSpectrum.stack.calc_line_flux_ratios(4975, dw=10, tag='test', save=False, conf=None, path='')
        self.assertEqual(len(out[0]), len(TestSpectrum.stack))
        self.assertTrue(_wl > 0)
        self.assertTrue(_wr > 0)


if __name__ == '__main__':
    unittest.main()
