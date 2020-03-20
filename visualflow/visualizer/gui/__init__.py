
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from tvtk.pyface.scene_model import SceneModel
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel


class TraitsDialog(HasTraits):
    """
    Graphic user interface implementation for visualflow, using traits.
    """
    scene = Instance(MlabSceneModel, ())

    view = View(Item('scene', height=400, show_label=False, editor=SceneEditor(scene_class=MayaviScene)))

