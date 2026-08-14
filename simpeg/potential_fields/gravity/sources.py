from ...survey import BaseSrc


class SourceField(BaseSrc):
    # docerator: override=receiver_list
    """Source field for gravity integral formulation

    Parameters
    ----------
    receiver_list : list of simpeg.potential_fields.gravity.receivers.Point
        List of gravity receivers
    """

    def __init__(self, receiver_list=None, **kwargs):
        super(SourceField, self).__init__(receiver_list=receiver_list, **kwargs)
