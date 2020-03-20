import sys
import os
import time

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_DIR)
DATASET_DIR = os.path.join( PROJECT_DIR,'data')


def test_integration():
    from visualflow.dataloader.dataset import Dataset

    ds = Dataset(os.path.join(DATASET_DIR, '45x30x20', '3D-V.dat'))
    #ds = Dataset(os.path.join(DATASET_DIR, '100x60x30', '3D-V.dat'))

    ds.find_all_criticals()

    ds.generate_seeds()

    ds.render_streamline()


# def test_gui():
#     from visualflow.visualizer.gui import TraitsDialog
#
#     gui = TraitsDialog()
#
#     test_integration(gui.scene.mayavi_scene)
#
#     gui.configure_traits()


if __name__ == '__main__':
    t = time.time()

    test_integration()

    print(time.time() - t)