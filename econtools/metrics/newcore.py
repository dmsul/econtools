from __future__ import division

from econtools.util import force_list
from regutils import unpack_spatialargs


def reg(df, y_name, x_name,
        a_name=None, nosingles=True,
        vce_type=None, cluster=None, spatial_hac=None,
        addcons=None, nocons=False,
        awt_name=None
        ):
    pass


def ivreg(df, y_name, x_name, z_name, w_name,
          a_name=None, nosingles=True,
          iv_method='2sls', _kappa_debug=None,
          vce_type=None, cluster=None, spatial_hac=None,
          addcons=None, nocons=False,
          awt_name=None,
          ):
    pass


class RegressionBase(object):

    def __init__(self, df, y_name, x_name, **kwargs):
        self.df = df
        self.y_name = y_name
        self.__dict__.update(kwargs)

        self.sample_cols = (
            'y_name', 'x_name', 'a_name', 'cluster', 'spatial_x', 'spatial_y',
            'awt_name'
        )

        # Set `vce_type`
        self.vce_type = _set_vce_type(self.vce_type, self.cluster,
                                      self.spatial_hac)
        # Unpack spatial HAC args
        sp_args = unpack_spatialargs(self.spatial_hac)
        self.spatial_x = sp_args[0]
        self.spatial_y = sp_args[1]
        self.spatial_band = sp_args[2]
        self.spatial_kern = sp_args[3]
        # Force variable names to lists
        self.x_name = force_list(x_name)

    def set_sample(self):
        pass


class Regression(RegressionBase):

    def __init__(self, *args, **kwargs):
        super(Regression, self).__init__(*args, **kwargs)

    def set_sample(self):
        sample_cols = tuple([self.__dict__[x] for x in self.sample_cols])
        sample = flag_sample(df, *sample_cols)
        if nosingles and a_name:
            sample &= flag_nonsingletons(df, a_name, sample)
        y, x, A, cluster_id, space_x, space_y, w, z, AWT = set_sample(
            df, sample, sample_cols)


class IVReg(RegressionBase):

    def __init__(self, df, y_name, x_name, z_name, w_name, **kwargs):
        super(Regression, self).__init__(df, y_name, x_name, **kwargs)
        # Handle extra variable stuff for IV
        self.z_name = force_list(z_name)
        self.w_name = force_list(w_name)
        self.sample_cols += ('z_name', 'w_name')


def _set_vce_type(vce_type, cluster, spatial_hac):
    """ Check for argument conflicts, then set `vce_type` if needed.  """
    # Check for valid arg
    valid_vce = (None, 'robust', 'hc1', 'hc2', 'hc3', 'cluster', 'spatial')
    if vce_type not in valid_vce:
        raise ValueError("VCE type '{}' is not supported".format(vce_type))
    # Check for conflicts
    cluster_err = cluster and (vce_type != 'cluster' and vce_type is not None)
    shac_err = spatial_hac and (vce_type != 'spatial' and vce_type is not None)
    if (cluster and spatial_hac) or cluster_err or shac_err:
        raise ValueError("VCE type conflict!")
    # Set `vce_type`
    if cluster:
        new_vce = 'cluster'
    elif spatial_hac:
        new_vce = 'spatial'
    else:
        new_vce = vce_type

    return new_vce


if __name__ == '__main__':
    pass
