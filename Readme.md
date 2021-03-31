Blender addon that takes a mesh object and creates a parent hierarchy of objects, one for each face, where each object's local X axis lies along the edge it shares with its parent. Works on closed shapes by creating a net and kinda works on actual origami shapes.

Each object's local axis are aligned as so:

  - X lies along the hinge edge (edge it shares with its parent)
  - Y lies on the face (ie. a small step in +Y would land you on the face, whereas a small step in -Y would land in space)
  - Z is along the face normal
  - X is oriented such that positive X rotation causes the face to move in the direction of the face normal (ie right hand rule)

[Demo](https://youtu.be/28FP0Ip6860)

Available in the menu Object>Orgamify or by operator search.

## Installation

Unzip and select the file `origami.py` when installing from Blender, not the `.zip` file.


## Known Issues

  - Some objects are created with a flipped normal. Cause unknown, but the axes created due respect the face normal from the original object
