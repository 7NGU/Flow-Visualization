
import re
import numpy as np
import os
from scipy import optimize, linalg
import logging
import json_tricks as json
from mayavi import mlab
from visualizer.core.pysac.mayavi_seed_streamlines import SeedStreamline
from dataloader.entropyfield import EntropyField
from random import random
from localconfig import LocalConfig


# config logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)


class Dataset(object):
    def __init__(self, filename):
        self.filename = filename
        
        # load configurations
        self.config = LocalConfig(os.path.join(os.path.dirname(self.filename), 'config.ini'))

        self.field_names = ['x', 'y', 'z', 'head', 'vx', 'vy', 'vz', 'v']
        self.fields = {}
        self.axis_ranges = {}
        self.axis_width = {}
        self.resolution_level = self.config.template_seeds.seed_plane_to_critical
        self.resolution = None

        self._load()

        # critical points info
        self.load_criticals_from_file = self.config.criticals.load_criticals_from_file
        self.criticals_info = []

        # seeds
        self.seeds = None
        self.seeded = np.zeros(self.grid_size, dtype='bool')
        self.seed_per_critical_plane = self.config.template_seeds.seeds_per_plane

        # calculate entropy
        self.entropy_field = self._calculate_entropy()
        self.entropies = self.entropy_field.entropies
        self.max_entropy = self.entropies.max()
        self.entropy_factor = self.config.entropy.entropy_threshold
        # enable to skip low entropy grid for critical searching
        self.entropy_filter = True
        self.region_of_interest = (self.entropies >= self.entropy_factor * self.max_entropy)

        # overall seeding
        self.target_seeds_number = int(np.prod(self.grid_size) * self.config.overall_seeding.seeding_frequency)
        self.mode = self.config.overall_seeding.mode

    def generate_seeds(self):
        """
        Generates seeds.
        1) Template seeds.
        2) Seeds in high entropy grids, or uniform seeds, depending on mode.
        """
        logger.info("Generating seeds for the whole dataset.")

        self._generate_template_seeds(self.seed_per_critical_plane)

         #visualize template seeds
         #mlab.points3d(*self.seeds.T, color=(1, 0, 0), scale_factor=0.002, opacity=0.5)

        seeds_needed_num = self.target_seeds_number - self.seeds.shape[0]
        unseeded_grids_num = (1 - self.seeded).sum()
        unseeded_important_grids_num = ((1 - self.seeded) & self.region_of_interest).sum()

        if self.mode == 'entropy':
            self._generate_entropy_seeds(seeds_needed_num / unseeded_important_grids_num)
        elif self.mode == 'uniform':
            self._generate_uniform_seeds(seeds_needed_num / unseeded_grids_num)
        elif self.mode == 'both':
            ratio = self.config.both_mode_setting.entropy_seeding_ratio
            self._generate_entropy_seeds(ratio * seeds_needed_num / unseeded_important_grids_num)

            seeds_needed_num = self.target_seeds_number - self.seeds.shape[0]
            unseeded_grids_num = (1 - self.seeded).sum()
            self._generate_uniform_seeds(seeds_needed_num / unseeded_grids_num)
        else:
            pass

    def query_by_index(self, i, j, k, query_list='all'):
        """
        Returns desired sample data, according to index of x, y, z coordinates.
        :param i:           <int> or slice      index of x-axis
        :param j:           <int> or slice      index of y-axis
        :param k:           <int> or slice      index of z-axis
        :param query_list:  'all' or <list>     e.g. ['x', 'y', 'z', 'vx', 'vy', 'vz']
        :return:            <dict>
        """
        if query_list == 'all':
            query_list = self.field_names

        result = {}
        for query in query_list:
            result[query] = self.fields[query][i, j, k]

        return result

    def find_all_criticals(self, criticals_file=None):
        """
        Finds all critical points inside the dataset grids. Updates `self.criticals_info`
        :param criticals_file:      <str>       critical points information filename
        """
        if self.load_criticals_from_file:
            self._load_critical_points(filename=criticals_file)
            return

        logger.info("Searching for critical points...")

        count = 0
        for i in range(self.grid_size[0]):
            logger.info("%d/%d grids searched." % (i * self.grid_size[1] * self.grid_size[2],
                                                   np.prod(self.grid_size)))

            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    ok, critical, jac = self._find_critical(i, j, k)
                    if ok:
                        count += 1

                        eigvals = linalg.eigvals(jac)
                        infodict = {
                            'grid_index': [i, j, k],
                            'pos': critical,
                            'jac': jac,
                            'eigvals': eigvals,
                            'seeding_template': self._get_seeding_template(eigvals)
                        }

                        self.criticals_info.append(infodict)
                        logger.info("%d criticals found." % count)

        logger.info("Complete searching critical points, %d found." % count)

        if not self.load_criticals_from_file:
            self._save_critical_points(criticals_file)

    def render_streamline(self):
        """
        Renders streamlines to mayavi.mlab
        """
        params = [self.fields[field] for field in ['x', 'y', 'z', 'vx', 'vy', 'vz']]

        # mlab.figure()

        field = mlab.pipeline.vector_field(*params, scalars=self.fields['v'])
        streamlines = SeedStreamline(seed_points=self.seeds)
        streamlines.stream_tracer.integration_direction = 'both'
        streamlines.stream_tracer.initial_integration_step = 0.02
        field.add_child(streamlines)

        mlab.scalarbar(streamlines, title='Velocity', orientation='vertical')
        mlab.title('Mode:' + self.mode)
        mlab.outline()
        mlab.show()

    def _load(self):
        """
        Loads data from file.

        e.g. filename='3D-V.dat'

        VARIABLES ="X","Y","Z","Head","Vx","Vy","Vz","Vsum"
        ZONE I=45, J=30, K=20, DATAPACKING=POINT
        0.000000 0.000000 0.000000 2.000000 0.000000 0.000000 -0.408407 0.408407
        0.022727 0.000000 0.000000 1.968146 0.109875 0.000000 -0.381347 0.396860
        0.045455 0.000000 0.000000 1.880071 0.193943 0.000000 -0.306749 0.362917
        0.068182 0.000000 0.000000 1.756408 0.232707 0.000000 -0.202748 0.308641
        0.090909 0.000000 0.000000 1.625893 0.217740 0.000000 -0.094596 0.237401
        ...

        """
        logger.info("Loading data from %s" % self.filename)

        with open(self.filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    # first line
                    pass
                elif i == 1:
                    # second line
                    a, b, c = map(int, re.findall(r'=(\d+),', line))
                    self.size = [a, b, c]
                    self.grid_size = [a-1, b-1, c-1]
                    self.data = np.empty([a*b*c, 8])
                else:
                    # data lines
                    self.data[i - 2, :] = [float(num) for num in line.split()]

        def _structure_data(col):
            return col.reshape(self.size, order='F')

        for i, name in enumerate(self.field_names):
            self.fields[name] = _structure_data(self.data[:, i])

        for i, axis in enumerate(['x', 'y', 'z']):
            start = self.data[0, i]
            end = self.data[-1, i]
            if start > end:
                # re-order data
                self.axis_ranges[axis] = [end, start]
                self.axis_width[axis] = start - end
                for k in ['x', 'y', 'z']:
                    self.fields[k] = np.flip(self.fields[k], i)
                    self.fields['v' + k] = np.flip(self.fields['v' + k], i)
                # flip v
                self.fields['v'] = np.flip(self.fields['v'], i)
            else:
                self.axis_ranges[axis] = [start, end]
                self.axis_width[axis] = end - start

        self.resolution = np.divide(list(self.axis_width.values()), self.grid_size).min() * self.resolution_level

        logger.info("Data loaded.")

    def _has_critical_point(self, i, j, k):
        """
        Judges whether there exists a critical point in the given grid.
        8 grid vertices are defined as:
        (x[i], x[i+1]) X (y[j], y[j+1]) X (z[k], z[k+1])
        """
        if not self.region_of_interest[i, j, k]:
            return False

        vx = self.fields['vx'][i:i+2, j:j+2, k:k+2]
        vy = self.fields['vy'][i:i+2, j:j+2, k:k+2]
        vz = self.fields['vz'][i:i+2, j:j+2, k:k+2]

        if (vx > 0).all() and (vy > 0).all() and (vz > 0).all():
            return False
        elif (vx < 0).all() and (vy < 0).all() and (vz < 0).all():
            return False

        return True

    def _find_critical(self, i, j, k):
        """
        Finds the critical point inside the (i,j,k) grid.
        :param i:       <int>       index of x-axis
        :param j:       <int>       index of y-axis
        :param k:       <int>       index of z-axis
        :return:        (ok, solution, fjac)
        """
        if not self._has_critical_point(i, j, k):
            return False, None, None

        axes = ['x', 'y', 'z']
        data = {}

        for idx, axis in enumerate(axes):
            v_name = 'v' + axis

            axis_data = self.fields[axis][i:i + 2, j:j + 2, k:k + 2].ravel()
            start, end = axis_data[0], axis_data[-1]

            data[axis] = [start, end]
            data[v_name] = self.fields[v_name][i:i + 2, j:j + 2, k:k + 2]

        def _equations(pos):
            x, y, z = pos

            def _predict(v_data):
                xd = (x - data['x'][0]) / (data['x'][1] - data['x'][0])
                yd = (y - data['y'][0]) / (data['y'][1] - data['y'][0])
                zd = (z - data['z'][0]) / (data['z'][1] - data['z'][0])

                c00 = v_data[0, 0, 0] * (1 - xd) + v_data[1, 0, 0] * xd
                c01 = v_data[0, 0, 1] * (1 - xd) + v_data[1, 0, 1] * xd
                c10 = v_data[0, 1, 0] * (1 - xd) + v_data[1, 1, 0] * xd
                c11 = v_data[0, 1, 1] * (1 - xd) + v_data[1, 1, 1] * xd

                c0 = c00 * (1 - yd) + c10 * yd
                c1 = c01 * (1 - yd) + c11 * yd

                return c0 * (1 - zd) + c1 * zd

            return [_predict(data[name]) for name in ['vx', 'vy', 'vz']]

        grid_center = [np.mean(data[axis]) for axis in axes]

        sol, infodict, ier, mesg = optimize.fsolve(_equations, grid_center, full_output=True)

        if ier == 1:
            # a solution is found
            if data['x'][0] <= sol[0] <= data['x'][1] and data['y'][0] <= sol[1] <= data['y'][1] and \
                    data['z'][0] <= sol[2] <= data['z'][1]:
                # solution inside the given grid
                logger.info("Function evaluated at the output: %s" % infodict['fvec'])
                return True, sol, infodict['fjac']

        return False, None, None

    def _save_critical_points(self, filename=None):
        """
        Saves critical points to json file.
        :param filename:        <str>
        """
        logger.info("Saving critical points info to json file...")

        if filename is None:
            filename = os.path.join(os.path.dirname(self.filename), 'criticals.json')

        with open(filename, 'w') as f:
            json.dump(self.criticals_info, f)

        logger.info("%s saved." % filename)

    def _load_critical_points(self, filename=None):
        """
        Loads critical points from json file.
        :param filename:        <str>
        """
        logger.info("Loading critical points from file...")

        if filename is None:
            filename = os.path.join(os.path.dirname(self.filename), 'criticals.json')

        with open(filename, 'r') as f:
            self.criticals_info = json.load(f, preserve_order=False)

        logger.info("Critical points loaded.")

    def _get_seeding_template(self, eigvals):
        """
        Gets the seeding template for the three given eigenvalues.
        :param eigvals:     <np.array, shape=(3,)>
        :return:            <str>       e.g. 'a'

        Template list: (Note: there must exist at least one real eigenvalue out of three eigenvalues)
        template_code   name                eigenvalues
        'a'             source              all positive real
        'a'             sink                all negative real
        'b'             spiral source       1 positive real, 2 positive real parts
        'b'             spiral sink         1 negative real, 2 negative real parts
        'c'             spiral saddle       1 positive real, 2 negative real parts
        'c'             spiral saddle       1 negative real, 2 positive real parts
        'd'             saddle              1 positive real, 2 negative real
        'd'             saddle              1 negative real, 2 positive real
        """
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        if np.all(imag_parts == 0):
            # all real
            if np.all(real_parts > 0) or np.all(real_parts < 0):
                return 'a'

            if np.count_nonzero(real_parts > 0) == 1 or np.count_nonzero(real_parts < 0) == 1:
                return 'd'

        if np.count_nonzero(imag_parts == 0) == 1:
            # only one real
            if np.all(real_parts > 0) or np.all(real_parts < 0):
                return 'b'

            if np.count_nonzero(real_parts > 0) == 1 or np.count_nonzero(real_parts < 0) == 1:
                return 'c'

        return 'unexpected'

    def _template_seeds(self, template_name, pos, jac, seed_to_critical, seed_num, seed_to_seed=None):
        """
        Returns seeds for the given template at the given critical point.
        :param template_name:       <str>                           e.g. 'a'
        :param pos:                 <np.array, shape=(3,)>          x,y,z coordinates of the critical point
        :param jac:                 <np.array, shape=(3,3)>         Jacobian matrix at the critical point
        :param seed_to_critical:    <float>                         distance from critical point to the seeding plane
        :param seed_num:            <int>                           number of seeds on each seeding plane
        :param seed_to_seed:        <float>                         distance from seed to seed, or radius for some templates
        :return:                    <np.array, shape=(seed_num,3)>  x,y,z coordinates of generated seeds
        """
        if seed_to_seed is None:
            seed_to_seed = seed_to_critical

        # get plane centers
        normal_vecs = [0, 1, 2]          # x,y,z axis
        plane_base_vecs = [(1, 2), (2, 0), (0, 1)]       # (y,z), (z,x), (x,y)

        seeds = None

        for normal_vec, plane_base_pair in zip(normal_vecs, plane_base_vecs):

            base0, base1 = plane_base_pair
            plane_centers = [pos + seed_to_critical * jac[normal_vec], pos - seed_to_critical * jac[normal_vec]]

            xyz_prime = np.zeros((seed_num, 3))

            if template_name == 'a':
                r = seed_to_seed
                thetas = np.linspace(0, 2 * np.pi, seed_num, endpoint=False)
                x_prime = r * np.cos(thetas)
                y_prime = r * np.sin(thetas)

                xyz_prime[:, base0] = x_prime
                xyz_prime[:, base1] = y_prime
            elif template_name == 'b' or template_name == 'c':
                r = (seed_num - 1) * seed_to_seed / 2
                x_prime = np.linspace(-r, r, seed_num)

                xyz_prime[:, base0] = x_prime
            else:
                # template_name == 'd'
                r = seed_to_seed
                thetas = np.linspace(0, 2 * np.pi, seed_num, endpoint=False)
                x_prime = r * np.cos(thetas)
                y_prime = r * np.sin(thetas)

                xyz_prime1 = xyz_prime.copy()
                xyz_prime[:, base0] = x_prime
                xyz_prime[:, base1] = y_prime

                x_prime1 = r/2 * np.cos(thetas)
                y_prime1 = r/2 * np.sin(thetas)
                xyz_prime1[:, base0] = x_prime1
                xyz_prime1[:, base1] = y_prime1

                xyz_prime = np.vstack((xyz_prime, xyz_prime1))

            xyz = np.matmul(xyz_prime, jac)

            plane_seeds = np.vstack([xyz + center for center in plane_centers])

            if seeds is None:
                seeds = plane_seeds
            else:
                seeds = np.vstack((seeds, plane_seeds))

        return seeds

    def _calculate_entropy(self):
        """
        Calculates entropy of the vector field.
        :return:    <EntropyField>
        """
        return EntropyField(self.fields['vx'], self.fields['vy'], self.fields['vz'], self.fields['v'], 16)

    def _generate_template_seeds(self, seed_num):
        logger.info("Generating template seeds according to %d critical points." % len(self.criticals_info))
        seed_to_critical = self.resolution

        for info in self.criticals_info:
            seeds = self._template_seeds(info['seeding_template'], info['pos'], info['jac'], seed_to_critical, seed_num)

            if self.seeds is None:
                self.seeds = seeds
            else:
                self.seeds = np.vstack((self.seeds, seeds))

            i, j, k = info['grid_index']
            self.seeded[i, j, k] = True

        logger.info("%d seeds generated." % self.seeds.shape[0])

    def _generate_uniform_seeds(self, seed_frequency):
        """
        Generates uniform seeds in the grids which are not seeded.
        :param seed_frequency:      <float>     seeding frequency
        """
        logger.info("Generating uniformly random seeds in un-seeded grids, with frequency %.2f" % seed_frequency)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    if self.seeded[i, j, k]:
                        continue

                    def _generate_one_seed():
                        # generate one seed
                        x = np.random.uniform(self.fields['x'][i, 0, 0], self.fields['x'][i + 1, 0, 0])
                        y = np.random.uniform(self.fields['y'][0, j, 0], self.fields['y'][0, j + 1, 0])
                        z = np.random.uniform(self.fields['z'][0, 0, k], self.fields['z'][0, 0, k + 1])
                        return np.array([x, y, z])
                    if random() <= seed_frequency < 1:
                        self.seeds = np.vstack((self.seeds, _generate_one_seed()))
                        self.seeded[i, j, k] = True
                    elif seed_frequency >= 1:
                        for _ in range(int(seed_frequency)):
                            self.seeds = np.vstack((self.seeds, _generate_one_seed()))
                        if random() <= seed_frequency - int(seed_frequency) < 1:
                            self.seeds = np.vstack((self.seeds, _generate_one_seed()))
                        self.seeded[i, j, k] = True

        logger.info("%d seeds generated." % self.seeds.shape[0])

    def _generate_entropy_seeds(self, seed_frequency):
        logger.info("Generating random seeds in un-seeded, high-entropy grids, with frequency %.2f" % seed_frequency)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    if self.seeded[i, j, k] or not self.region_of_interest[i, j, k]:
                        continue

                    def _generate_one_seed():
                        # generate one seed
                        x = np.random.uniform(self.fields['x'][i, 0, 0], self.fields['x'][i + 1, 0, 0])
                        y = np.random.uniform(self.fields['y'][0, j, 0], self.fields['y'][0, j + 1, 0])
                        z = np.random.uniform(self.fields['z'][0, 0, k], self.fields['z'][0, 0, k + 1])
                        return np.array([x, y, z])

                    if random() <= seed_frequency < 1:
                        self.seeds = np.vstack((self.seeds, _generate_one_seed()))
                        self.seeded[i, j, k] = True
                    elif seed_frequency >= 1:
                        for _ in range(int(seed_frequency)):
                            self.seeds = np.vstack((self.seeds, _generate_one_seed()))
                        if random() <= seed_frequency - int(seed_frequency) < 1:
                            self.seeds = np.vstack((self.seeds, _generate_one_seed()))

                        self.seeded[i, j, k] = True

        logger.info("%d seeds generated." % self.seeds.shape[0])
