import isopy
import pytest
import isopy.toolbox as toolbox
import numpy as np

class Test_Inversion:
    def test_one(self):
        spike = isopy.array(pd104=1, pd106=0, pd108=1, pd110=0)
        spike = spike.normalise(1)

        self.compare_rudge_siebert('pd', spike, 1.6, 0.1, 0.5)
        self.compare_rudge_siebert('pd', spike, -1.6, -0.1, 0.85)
        self.compare_rudge_siebert('pd', spike, 1.6, -0.1, 0.15)

    def compare_rudge_siebert(self, element, spike, fins, fnat, spike_fraction):
        # Default reference values
        measured = isopy.tb.make_ms_sample(element, fnat=fnat, fins=fins,
                                           spike=spike, spike_fraction=spike_fraction,
                                           random_seed=46)

        result_rudge = isopy.tb.ds_inversion(measured, spike)
        result_siebert = isopy.tb.ds_inversion(measured, spike, method='siebert')


        assert type(result_rudge) is toolbox.doublespike.DSResult
        assert result_rudge.method == 'rudge'
        np.testing.assert_allclose(result_rudge.alpha, fnat*-1, rtol=0.1)
        np.testing.assert_allclose(result_rudge.beta, fins, rtol=0.1)
        np.testing.assert_allclose(result_rudge.fnat, fnat, rtol=0.1)
        np.testing.assert_allclose(result_rudge.fins, fins, rtol=0.1)
        np.testing.assert_allclose(result_rudge.spike_fraction, spike_fraction, rtol=1E-3)
        np.testing.assert_allclose(result_rudge.sample_fraction, (1-spike_fraction), rtol=1E-3)
        np.testing.assert_allclose(result_rudge.Q, (1 - spike_fraction)/spike_fraction, rtol=1E-3)

        assert type(result_siebert) is toolbox.doublespike.DSResult
        assert result_siebert.method == 'siebert'
        np.testing.assert_allclose(result_siebert.alpha, fnat * -1, rtol=0.1)
        np.testing.assert_allclose(result_siebert.beta, fins, rtol=0.1)
        np.testing.assert_allclose(result_siebert.fnat, fnat, rtol=0.1)
        np.testing.assert_allclose(result_siebert.fins, fins, rtol=0.1)
        np.testing.assert_allclose(result_siebert.spike_fraction, spike_fraction, rtol=1E-3)
        np.testing.assert_allclose(result_siebert.sample_fraction, (1 - spike_fraction), rtol=1E-3)
        np.testing.assert_allclose(result_siebert.Q, (1 - spike_fraction) / spike_fraction, rtol=1E-3)

        np.testing.assert_allclose(result_siebert.alpha, result_rudge.alpha, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.beta, result_rudge.beta, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.fnat, result_rudge.fnat, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.fins, result_rudge.fins, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.spike_fraction, result_rudge.spike_fraction)
        np.testing.assert_allclose(result_siebert.sample_fraction, result_rudge.sample_fraction)
        np.testing.assert_allclose(result_siebert.Q, result_rudge.Q)
        np.testing.assert_allclose(result_siebert.lambda_, result_rudge.lambda_)

        # Different reference values
        mass_ref = isopy.refval.isotope.mass_number
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        measured = isopy.tb.make_ms_sample(element, fnat=fnat, fins=fins,
                                           spike=spike, spike_fraction=spike_fraction,
                                           random_seed=46, isotope_masses=mass_ref,
                                           isotope_fractions=fraction_ref)

        result_rudge = isopy.tb.ds_inversion(measured, spike, fraction_ref, mass_ref)
        result_siebert = isopy.tb.ds_inversion(measured, spike, fraction_ref, mass_ref, method='siebert')

        assert type(result_rudge) is toolbox.doublespike.DSResult
        assert result_rudge.method == 'rudge'
        np.testing.assert_allclose(result_rudge.alpha, fnat * -1, rtol=0.1)
        np.testing.assert_allclose(result_rudge.beta, fins, rtol=0.1)
        np.testing.assert_allclose(result_rudge.fnat, fnat, rtol=0.1)
        np.testing.assert_allclose(result_rudge.fins, fins, rtol=0.1)
        np.testing.assert_allclose(result_rudge.spike_fraction, spike_fraction, rtol=1E-3)
        np.testing.assert_allclose(result_rudge.sample_fraction, (1 - spike_fraction), rtol=1E-3)
        np.testing.assert_allclose(result_rudge.Q, (1 - spike_fraction) / spike_fraction, rtol=1E-3)

        assert type(result_siebert) is toolbox.doublespike.DSResult
        assert result_siebert.method == 'siebert'
        np.testing.assert_allclose(result_siebert.alpha, fnat * -1, rtol=0.1)
        np.testing.assert_allclose(result_siebert.beta, fins, rtol=0.1)
        np.testing.assert_allclose(result_siebert.fnat, fnat, rtol=0.1)
        np.testing.assert_allclose(result_siebert.fins, fins, rtol=0.1)
        np.testing.assert_allclose(result_siebert.spike_fraction, spike_fraction, rtol=1E-3)
        np.testing.assert_allclose(result_siebert.sample_fraction, (1 - spike_fraction), rtol=1E-3)
        np.testing.assert_allclose(result_siebert.Q, (1 - spike_fraction) / spike_fraction,
                                   rtol=1E-3)

        np.testing.assert_allclose(result_siebert.alpha, result_rudge.alpha, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.beta, result_rudge.beta, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.fnat, result_rudge.fnat, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.fins, result_rudge.fins, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.spike_fraction, result_rudge.spike_fraction)
        np.testing.assert_allclose(result_siebert.sample_fraction, result_rudge.sample_fraction)
        np.testing.assert_allclose(result_siebert.Q, result_rudge.Q)
        np.testing.assert_allclose(result_siebert.lambda_, result_rudge.lambda_)

    def test_result(self):
        spike = isopy.array(pd104=1, pd106=0, pd108=1, pd110=0)
        spike = spike.normalise(1)
        measured = isopy.tb.make_ms_sample('pd', fnat=0.1, fins=1.6,
                                           spike=spike, spike_fraction=0.5,
                                           random_seed=46)

        result = isopy.tb.ds_inversion(measured, spike)
        assert type(result) is toolbox.doublespike.DSResult
        attrs = 'alpha beta lambda_ fnat fins spike_fraction sample_fraction Q'.split()
        assert list(result.keys()) == attrs
        values = list(result.values())
        mean = np.mean(result)
        assert type(mean) is toolbox.doublespike.DSResult

        for i, (name, value) in enumerate(result.items()):
            assert name == attrs[i]
            assert result[name] is value
            assert value is getattr(result, name)
            assert value is values[i]
            assert getattr(mean, name) == np.mean(getattr(result, name))

        repr(result)

class Test_Correction:
    def test_one(self):
        spike = isopy.array(pd104=1, pd106=0, pd108=1, pd110=0)
        spike = spike.normalise(1)

        self.compare_rudge_siebert('pd', spike, 1.6, 0.1, 0.5)
        self.compare_rudge_siebert('pd', spike, -1.6, -0.1, 0.85)
        self.compare_rudge_siebert('pd', spike, 1.6, -0.1, 0.15)

        self.compare_rudge_siebert('pd', spike, 1.6, 0.1, 0.5, ru=0.01)
        self.compare_rudge_siebert('pd', spike, -1.6, -0.1, 0.85, cd=0.01)
        self.compare_rudge_siebert('pd', spike, 1.6, -0.1, 0.15, ru=0.01, cd=0.01)


    def compare_rudge_siebert(self, element, spike, fins, fnat, spike_fraction, **interferences):
        # Default reference values
        measured = isopy.tb.make_ms_sample(element, fnat=fnat, fins=fins,
                                           spike=spike, spike_fraction=spike_fraction,
                                           random_seed=46, **interferences)

        result_rudge = isopy.tb.ds_correction(measured, spike)
        result_siebert = isopy.tb.ds_correction(measured, spike, method='siebert')

        assert type(result_rudge) is toolbox.doublespike.DSResult
        assert result_rudge.method == 'rudge'
        np.testing.assert_allclose(result_rudge.alpha, fnat * -1, rtol=0.1)
        np.testing.assert_allclose(result_rudge.beta, fins, rtol=0.1)
        np.testing.assert_allclose(result_rudge.fnat, fnat, rtol=0.1)
        np.testing.assert_allclose(result_rudge.fins, fins, rtol=0.1)
        np.testing.assert_allclose(result_rudge.spike_fraction, spike_fraction, rtol=1E-3)
        np.testing.assert_allclose(result_rudge.sample_fraction, (1 - spike_fraction),
                                   rtol=1E-3)
        np.testing.assert_allclose(result_rudge.Q, (1 - spike_fraction) / spike_fraction,
                                   rtol=1E-3)

        assert type(result_siebert) is toolbox.doublespike.DSResult
        assert result_siebert.method == 'siebert'
        np.testing.assert_allclose(result_siebert.alpha, fnat * -1, rtol=0.1)
        np.testing.assert_allclose(result_siebert.beta, fins, rtol=0.1)
        np.testing.assert_allclose(result_siebert.fnat, fnat, rtol=0.1)
        np.testing.assert_allclose(result_siebert.fins, fins, rtol=0.1)
        np.testing.assert_allclose(result_siebert.spike_fraction, spike_fraction, rtol=1E-3)
        np.testing.assert_allclose(result_siebert.sample_fraction, (1 - spike_fraction),
                                   rtol=1E-3)
        np.testing.assert_allclose(result_siebert.Q, (1 - spike_fraction) / spike_fraction,
                                   rtol=1E-3)

        np.testing.assert_allclose(result_siebert.alpha, result_rudge.alpha, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.beta, result_rudge.beta, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.fnat, result_rudge.fnat, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.fins, result_rudge.fins, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.spike_fraction, result_rudge.spike_fraction)
        np.testing.assert_allclose(result_siebert.sample_fraction, result_rudge.sample_fraction)
        np.testing.assert_allclose(result_siebert.Q, result_rudge.Q)
        np.testing.assert_allclose(result_siebert.lambda_, result_rudge.lambda_)

        # Different reference values
        mass_ref = isopy.refval.isotope.mass_number
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        measured = isopy.tb.make_ms_sample(element, fnat=fnat, fins=fins,
                                           spike=spike, spike_fraction=spike_fraction,
                                           random_seed=46, isotope_masses=mass_ref,
                                           isotope_fractions=fraction_ref, **interferences)

        result_rudge = isopy.tb.ds_correction(measured, spike, isotope_fractions=fraction_ref, isotope_masses=mass_ref)
        result_siebert = isopy.tb.ds_correction(measured, spike, isotope_fractions=fraction_ref, isotope_masses=mass_ref,
                                               method='siebert')

        assert type(result_rudge) is toolbox.doublespike.DSResult
        assert result_rudge.method == 'rudge'
        np.testing.assert_allclose(result_rudge.alpha, fnat * -1, rtol=0.1)
        np.testing.assert_allclose(result_rudge.beta, fins, rtol=0.1)
        np.testing.assert_allclose(result_rudge.fnat, fnat, rtol=0.1)
        np.testing.assert_allclose(result_rudge.fins, fins, rtol=0.1)
        np.testing.assert_allclose(result_rudge.spike_fraction, spike_fraction, rtol=1E-3)
        np.testing.assert_allclose(result_rudge.sample_fraction, (1 - spike_fraction),
                                   rtol=1E-3)
        np.testing.assert_allclose(result_rudge.Q, (1 - spike_fraction) / spike_fraction,
                                   rtol=1E-3)

        assert type(result_siebert) is toolbox.doublespike.DSResult
        assert result_siebert.method == 'siebert'
        np.testing.assert_allclose(result_siebert.alpha, fnat * -1, rtol=0.1)
        np.testing.assert_allclose(result_siebert.beta, fins, rtol=0.1)
        np.testing.assert_allclose(result_siebert.fnat, fnat, rtol=0.1)
        np.testing.assert_allclose(result_siebert.fins, fins, rtol=0.1)
        np.testing.assert_allclose(result_siebert.spike_fraction, spike_fraction, rtol=1E-3)
        np.testing.assert_allclose(result_siebert.sample_fraction, (1 - spike_fraction),
                                   rtol=1E-3)
        np.testing.assert_allclose(result_siebert.Q, (1 - spike_fraction) / spike_fraction,
                                   rtol=1E-3)

        np.testing.assert_allclose(result_siebert.alpha, result_rudge.alpha, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.beta, result_rudge.beta, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.fnat, result_rudge.fnat, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.fins, result_rudge.fins, rtol=1E-6)
        np.testing.assert_allclose(result_siebert.spike_fraction, result_rudge.spike_fraction)
        np.testing.assert_allclose(result_siebert.sample_fraction, result_rudge.sample_fraction)
        np.testing.assert_allclose(result_siebert.Q, result_rudge.Q)
        np.testing.assert_allclose(result_siebert.lambda_, result_rudge.lambda_)


class Test_Delta:
    def test_delta(self):
        spike = isopy.array(pd104=1, pd106=0, pd108=1, pd110=0)
        spike = spike.normalise(1)
        measured = isopy.tb.make_ms_sample('pd', fnat=0.1, fins=1.6,
                                           spike=spike, spike_fraction=0.5,
                                           random_seed=46)

        result = isopy.tb.ds_inversion(measured, spike)

        mass_ratio1 = '108pd/105pd'
        mass_ratio2 = isopy.refval.isotope.mass_W17.get(mass_ratio1)

        correct = result.fnat
        correct1 = (np.power(mass_ratio2, correct) - 1)
        correct2 = np.log(mass_ratio2) * correct
        self.compare(correct1, correct2, mass_ratio1, result)
        self.compare(correct1, correct2, mass_ratio2, result)

        correct = result.fnat
        correct1 = (np.power(mass_ratio2, correct) - 1)
        correct2 = np.log(mass_ratio2) * correct
        self.compare(correct1, correct2, mass_ratio1, result)
        self.compare(correct1, correct2, mass_ratio2, result)

        correct = result.fnat
        correct1 = (np.power(mass_ratio2, correct) - 1)
        correct2 = np.log(mass_ratio2) * correct
        self.compare(correct1, correct2, mass_ratio1, result.fnat)
        self.compare(correct1, correct2, mass_ratio2, result.fnat)

        correct = 0 - np.mean(result.fnat)
        correct1 = (np.power(mass_ratio2, correct) - 1)
        correct2 = np.log(mass_ratio2) * correct
        self.compare(correct1, correct2, mass_ratio1, 0, result.fnat)
        self.compare(correct1, correct2, mass_ratio2, 0, result.fnat)

        correct = 0 - np.mean(result.fnat)
        correct1 = (np.power(mass_ratio2, correct) - 1)
        correct2 = np.log(mass_ratio2) * correct
        self.compare(correct1, correct2, mass_ratio1, 0, result)
        self.compare(correct1, correct2, mass_ratio2, 0, result)

        correct = 0 - (np.mean(result.fnat)/2 + 0.5)
        correct1 = (np.power(mass_ratio2, correct) - 1)
        correct2 = np.log(mass_ratio2) * correct
        self.compare(correct1, correct2, mass_ratio1, 0, (result.fnat, 1))
        self.compare(correct1, correct2, mass_ratio2, 0, (result.fnat, 1))

        correct = 0 - (np.mean(result.fnat) / 2 + 0.5)
        correct1 = (np.power(mass_ratio2, correct) - 1)
        correct2 = np.log(mass_ratio2) * correct
        self.compare(correct1, correct2, mass_ratio1, 0, (result, 1))
        self.compare(correct1, correct2, mass_ratio2, 0, (result, 1))

        mass_ratio1 = '108pd/105pd'
        mass_ratio2 = isopy.refval.isotope.mass_number.get(mass_ratio1)

        correct = result.fnat
        correct1 = (np.power(mass_ratio2, correct) - 1)
        correct2 = np.log(mass_ratio2) * correct
        self.compare(correct1, correct2, mass_ratio1, result, mass_ref=isopy.refval.isotope.mass_number)
        self.compare(correct1, correct2, mass_ratio2, result, mass_ref=isopy.refval.isotope.mass_number)


    def compare(self, correct, correct_prime, mass_ratio, fnat, reference_fnat=0, factor=1, mass_ref=None):
        result = isopy.tb.ds_Delta(mass_ratio, fnat, reference_fnat, factor=factor, isotope_masses=mass_ref)
        np.testing.assert_allclose(result, correct)

        result = isopy.tb.ds_Delta_prime(mass_ratio, fnat, reference_fnat, factor=factor, isotope_masses=mass_ref)
        np.testing.assert_allclose(result, correct_prime)