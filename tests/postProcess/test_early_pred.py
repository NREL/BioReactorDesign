from brd import BRD_EARLY_PRED_DATA_DIR
from brd.postProcess.early_pred import multi_data_load, fit_and_ext, bayes_fit, plotAllEarly, plotAllEarly_uq

def test_fit():
    data_dict, color_files = multi_data_load(BRD_EARLY_PRED_DATA_DIR)
    data_dict = fit_and_ext(data_dict)
    plotAllEarly(data_dict, color_files=color_files, chop=True, extrap=True)

def test_bayes_fit():
    data_dict, color_files = multi_data_load(BRD_EARLY_PRED_DATA_DIR)
    bayes_fit(data_dict)
    plotAllEarly_uq(data_dict, color_files=color_files)

