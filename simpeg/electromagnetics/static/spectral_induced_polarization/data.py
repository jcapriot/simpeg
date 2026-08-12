import numpy as np
from discretize.utils import mkvc

from ....data import Data as BaseData


class Data(BaseData):
    r"""
    Data class for spectral induced polarization data

    Parameters
    ----------
    survey : simpeg.survey.BaseSurvey
        A SimPEG survey object. For each geophysical method, the survey object defines
        the survey geometry; i.e. sources, receivers, data type.
    dobs : (n) numpy.ndarray
        Observed data.
    relative_error : None or float or numpy.ndarray, optional
        Assign relative uncertainties to the data using relative error; sometimes
        referred to as percent uncertainties. For each datum, we assume the
        standard deviation of Gaussian noise is the relative error times the
        absolute value of the datum; i.e. :math:`C_{err} \times |d|`.
    noise_floor : None or float or numpy.ndarray, optional
        Assign floor/absolute uncertainties to the data. For each datum, we assume
        standard deviation of Gaussian noise is equal to *noise_floor*.
    standard_deviation : None or float or numpy.ndarray, optional
        Directly define the uncertainties on the data by assuming we know the standard
        deviations of the Gaussian noise. This is essentially the same as *noise_floor*.
        If set however, this will override *relative_error* and *noise_floor*. If none
        are given, this defaults to 0.0
    """

    # docerator: provenance
    # docerator: from simpeg.data.Data: survey, dobs, relative_error, noise_floor, standard_deviation

    @property
    def index_dictionary(self):
        """
        Dictionary of data indices by sources and receivers. To set data using
        survey parameters:

        .. code::
            data = Data(survey)
            for src in survey.source_list:
                for rx in src.receiver_list:
                    for t in rx.times:
                        index = data.index_dictionary[src][rx][t]
                        data.dobs[index] = datum

        """
        if getattr(self, "_index_dictionary", None) is None:
            if self.survey is None:
                raise Exception(
                    "To set or get values by source-receiver pairs, a survey must "
                    "first be set. `data.survey = survey`"
                )

            # create an empty dict
            self._index_dictionary = {}

            # create an empty dict associated with each source
            for src in self.survey.source_list:
                self._index_dictionary[src] = {}

                for rx in src.receiver_list:
                    self._index_dictionary[src][rx] = {}

            # loop over sources and find the associated data indices
            indBot, indTop = 0, 0
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    for t in rx.times:
                        indTop += rx.nD
                        self._index_dictionary[src][rx][t] = np.arange(indBot, indTop)
                        indBot += rx.nD

        return self._index_dictionary

    ##########################
    # Methods
    ##########################

    def __setitem__(self, key, value):
        index = self.index_dictionary[key[0]][key[1]][key[2]]
        self.dobs[index] = mkvc(value)

    def __getitem__(self, key):
        index = self.index_dictionary[key[0]][key[1]][key[2]]
        return self.dobs[index]
