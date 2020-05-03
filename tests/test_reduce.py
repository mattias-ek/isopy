import isopy


def est_load_exp():
    isopy.read_exp('data/Pd_blank.exp')

def est_load_array():
    isopy.load_array('data/Pd_sample.csv', skip_columns=0)

def est_mass_independent():
    blank = isopy.read_exp('data/Pd_blank.exp')[1].isotope_data
    sample = isopy.load_array('data/Pd_sample.csv', skip_columns=0)
    answers = isopy.load_array('data/Pd_answers.csv', skip_columns=0)

    assert len(blank) == 60

    blk_avg = isopy.np.mean(blank)
    assert_array_equals(blk_avg, answers[0], 9)

    smp_avg = isopy.np.mean(sample)
    smp_sd = isopy.sd(sample)
    smp_median = isopy.np.median(sample)
    smp_mad = isopy.mad(sample)

    outliers = isopy.toolbox.isotope.find_outliers_mad(sample, axis=0)
    sample2 = isopy.toolbox.isotope.replace_outliers(sample, outliers)

    smp_avg2 = isopy.np.nanmean(sample2)
    smp_sd2 = isopy.nansd(sample2)
    smp_median2 = isopy.np.nanmedian(sample2)
    smp_mad2 = isopy.nanmad(sample2)

    sample3 = sample2 - blk_avg


def assert_array_equals(array1, array2, decimals):
    for k in array1.keys():
        assert isopy.np.round(array1[k], decimals) == isopy.np.round(array2[k], decimals)

