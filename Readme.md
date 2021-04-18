Blender addon that takes a mesh object and creates a parent hierarchy of objects, one for each face, where each object's local X axis lies along the edge it shares with its parent. Works on closed shapes by creating a [net](https://en.wikipedia.org/wiki/Net_(polyhedron)) and kinda works on actual origami shapes.

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

To use the `Unfold`, `Fold`, and `Animate` commands, you need to first run the `Origamify` command on the object you're interested in. Each of the commands then operates on the created parent hierarchy (or some subset of it).

Objects that have been run through the `Origamify` command will have some custom properties on them which start with `origami_`.

### Origamify

Create the parent hierarchy from a selected mesh with the axes as explained above. You can control the root object by going into Edit Mode, selecting that face, back into Object Mode, and then running this command.

The constrain root option inserts an X Limit Rotation with min and max angle of 0, which locks the X axis rotation for the parent. This is just a convenience because if you `Select Hierachy` and rotate all objects on their local X, things will be weird if you also rotate the root.

### Origami Unfold

This sets the local X rotation of each selected object (does not recurse the hierarchy, though this might be a nice-to-have in the future) such that it is coplanar and non-overlapping with its parent.

### Origami Fold

This sets the local X rotation of each selected object to the value that restores it to the original mesh's rotation.

### Origami Animate

Select just the object you want to animate and this descends recursively to insert keyframes for the fold or unfold rotations.

## Known Issues

  - Some objects are created with a flipped normal. Cause unknown, but the axes created do respect the face normal from the original object

## Tips

Because each object's local X axis is its "hinge" axis to its parent, it is very useful to select multiple objects you're interested in rotating, then holding `Alt` and dragging the `Rotation X` in `Object Properties` or holding `Alt` and clicking into `Rotation X` and entering a value.

## Notes

I originally wrote this wanting to achieve something like [this](https://origamisimulator.org/), where given a flat net you could set the face normals for the direction of fold and get control of the fold with the parent hierarchy and the local X rotation. That is not feasible and will never work with this addon.

Flat nets (meshes where all faces are coplanar) can still be processed with this tool to create the parent hierarchy, but note that the `Fold` and `Unfold` commands no longer make sense because the mesh is already flat and its original position is flat.

The net that is generated is not guaranteed to be collision free everywhere.

## Related Tools

  - <https://origamisimulator.org/>
  - <https://docs.blender.org/manual/en/latest/addons/import_export/paper_model.html>
  - <https://tamasoft.co.jp/pepakura-en/>
