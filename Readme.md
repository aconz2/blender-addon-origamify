Blender addon that takes a mesh object and creates a parent hierarchy of objects, one for each face, where each object's local X axis lies along the edge it shares with its parent. Works on closed shapes by creating a [net](https://en.wikipedia.org/wiki/Net_(polyhedron) and kinda works on actual origami shapes.

Each object's local axis is aligned so that:

  - X lies along the hinge edge (edge it shares with its parent)
  - Y lies on the face (ie. a small step in +Y would land you on the face, whereas a small step in -Y would land in space)
  - Z is along the face normal
  - X is oriented such that positive X rotation causes the face to move in the direction of the face normal (ie right hand rule)

[Demo](https://youtu.be/28FP0Ip6860)

Available in the menu Object>Orgamify or by operator search.

## Installation

Unzip and select the file `origami.py` when installing from Blender, not the `.zip` file.

## Commands

### Origamify

Create the parent hierarchy from a selected mesh with the axes as explained above. You can control the root object by going into Edit Mode, selecting that face, back into Object Mode, and then running this command.

### Origami Unfold

This sets the local X rotation of each selected object (does not recurse the hierarchy, though this might be a nice-to-have in the future) such that it is coplanar and non-overlapping with its parent.

### Origami Fold

This sets the local X rotation of each selected object to the value that restores it to the original mesh's rotation.

### Origami Animate

Select just the object you want to animate and this descends recursively to insert keyframes for the fold or unfold rotations.

## Known Issues

  - Some objects are created with a flipped normal. Cause unknown, but the axes created due respect the face normal from the original object

## Notes

I originally wrote this wanting to achieve something like [this](https://origamisimulator.org/), where given a flat net you could set the face normals for the direction of fold and get control of the fold with the parent hierarchy and the local X rotation. That is not feasible and will never work with this addon.

Flat nets (meshes where all faces are coplanar) can still be processed with this tool to create the parent hierarchy, but note that the `Fold` and `Unfold` commands no longer make sense because the mesh is already flat and its original position is flat.

The net that is generated is not guaranteed to be collision free everywhere.

## Related Tools

  - <https://origamisimulator.org/>
  - <https://docs.blender.org/manual/en/latest/addons/import_export/paper_model.html>
  - <https://tamasoft.co.jp/pepakura-en/>
