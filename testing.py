import isopy
import numpy as np
import datetime


if __name__ == "__main__":

    ref_data = isopy.read.reference_isotope_data()

    np.seterr(all='ignore')

    max_voltage = [10]
    resolution = 49
    cycles = 60
    integration_time = 4
    isotopes = ['Ge70', 'Ge72', 'Ge73', 'Ge74', 'Ge76']
    #isotopes = ['Pd102', 'Pd104', 'Pd105', 'Pd106', 'Pd108', 'Pd110']

    for m_v in max_voltage:
        continue
        result = isopy.tb.doublespike.cocktail_list(ref_data['initial isotope fraction L09'], ref_data['isotope mass H17'],
                                           isotopes = isotopes, output_mass_ratio = (74/70), output_multiple = 1000,
                                           max_voltage=m_v, resolution=resolution, cycles=cycles,
                                           integration_time=integration_time)

        for res in result:
            print('Inverstion: {}, Spike: {}, Smallest uncertianty:, {:.5f}, at x:, {:.2f}, y:, {:.2f}'.format(
                '-'.join(res[0]), '-'.join(res[1]), res[2][2], res[2][0], res[2][1]))


    x, y , z, best = isopy.tb.doublespike.plot_uncertianty_grid(ref_data['initial isotope fraction L09'], 'Ge70', 'Ge73',
                                                       ref_data['isotope mass H17'],
                                                       isotopes = ['Ge70', 'Ge72', 'Ge73', 'Ge74'], max_voltage = 10,
                                                       resolution = 99, integration_time = 4)

    x, y, z, best = isopy.tb.doublespike.plot_uncertianty_grid(ref_data['initial isotope fraction L09'], 'Ge73', 'Ge76',
                                                          ref_data['isotope mass H17'],
                                                          isotopes=['Ge70', 'Ge73', 'Ge74', 'Ge76'], max_voltage=10,
                                                          resolution=99, integration_time=4)


