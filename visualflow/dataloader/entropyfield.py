
import numpy as np
from math import sqrt, log
import collections
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)


class EntropyField(object):
    """
    Computes entropy of a given vector field.
    """

    def __init__(self, vx, vy, vz, v, bin_count):
        """
        Constructor.
        :param vx:      x-projection of vector field
        :param vy:      y-projection of vector field
        :param vz:      z-projection of vector field
        :param v:       norm of each vector
        :param bin_count:   number of bins, should be a perfect square integer.
        """
        # cartesian coordinates
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.v = v

        # number of bins, to calculate entropy
        self.bin_count = bin_count
        # stores the bin index (from 0 to bin_count-1) of each vector
        self.bin_map = self._map_to_bin()

        # entropy of each grid
        self.grid_size = [d - 1 for d in self.v.shape]
        self.entropies = np.zeros(self.grid_size)
        self._calculate_entropy()

    def _map_to_bin(self):
        """
        Maps each vector into a corresponding bin.
        :return:    bin_map
        """
        logger.info("Mapping each vector direction into %d bins ..." % self.bin_count)

        # normalize to unit vectors
        vx = self.vx / self.v
        vy = self.vy / self.v
        vz = self.vz / self.v

        # reference: http://mathworld.wolfram.com/Zone.html
        # Fact: the surface area of a spherical segment, which is bordered by two parallel latitudes,
        # depends only on the height of the zone.

        # So we evenly divide the theta (angle in xy-plane with respect to axis-x) into N partitions,
        # i.e. generating N uniform longitudes.
        # Then we evenly divide the z-axis into N partitions,
        # i.e. generation N strips with same areas.
        # Finally we got N**2 equal-area surface zones in total.
        N = int(sqrt(self.bin_count))

        # range of arctan2: [-pi, pi]
        longitudes = np.arctan2(vy, vx) + np.pi
        bin_width = 2 * np.pi / N
        # i-index of bin
        bin_i = np.floor(longitudes / bin_width).astype(int)
        np.clip(bin_i, 0, N - 1, bin_i)

        # range of normalized vz: [-1, 1]
        elevations = vz + 1
        bin_width = 2 / N
        # j-index of bin
        bin_j = np.floor(elevations / bin_width).astype(int)
        np.clip(bin_j, 0, N - 1, bin_j)

        bin_map = np.ravel_multi_index((bin_i, bin_j), (N, N))

        return bin_map

    def _calculate_entropy(self):
        """
        Calculates entropy of each grid.
        Updates self.entropies
        """
        logger.info("Calculating entropy of each grid ...")

        v_per_grid = 8
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    grid_bin_map = self.bin_map[i:i+2, j:j+2, k:k+2]

                    counter = collections.Counter(grid_bin_map.ravel())

                    for count in counter.values():
                        p = count / v_per_grid
                        try:
                            self.entropies[i, j, k] += -p * log(p)
                        except ValueError as e:
                            pass
