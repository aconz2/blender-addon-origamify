
bl_info = {
    'name': 'Origamify',
    'description': 'Create a parent hierarchy, one object per face, to unwrap objects like origami',
    'blender': (2, 80, 0),
    'category': 'Object',
}

import bpy
import bmesh
from mathutils import Vector, Matrix, Euler
import math
from collections import defaultdict
from queue import deque
import random
import itertools

def rotate_about_axis(axis, theta):
    """
    rodrigues formula
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis.normalized()
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return Matrix([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def change_of_basis_matrix(at, i, j, k):
    rot = Matrix([i.normalized(), j.normalized(), k.normalized()])
    return Matrix.Translation(at) @ rot.transposed().to_4x4()

def canonical_key(face1, edge, face2):
    if face1 < face2:
        return (face1, edge, face2)
    return (face2, edge, face1)

def face_connectivity_graph(mesh):
    """
    Returns a list of triples (face_idx0, edge_idx, face_idx1) which correspond to the linkage information
    face_idx0 < face_idx1 to dedup
    raises Exception if edge connects more than two faces
    """
    mesh.faces.ensure_lookup_table()
    mesh.edges.ensure_lookup_table()

    ret = set()
    for face in mesh.faces:
        for loop in face.loops:
            faces = list(loop.edge.link_faces)
            if len(faces) > 2:
                raise Exception('Edges connect more than two faces')
            if len(faces) == 2:
                ret.add(canonical_key(faces[0].index, loop.edge.index, faces[1].index))
            # ignore edges on boundaries if not a closed shape for instance

    return list(ret)

def spanning_tree(edges, start=None, breadthfirst=False):
    """
    Computes an arbitrary spanning tree from a list of triples (face_idx0, edge_idx, face_idx1)
    Returns a list which is a subset of the input
    """
    # each edge only appears once
    assert len(set(e for _, e, _ in edges)) == len(edges)

    g = defaultdict(list)
    faces = set()

    for f1, e, f2 in edges:
        g[f1].append((e, f2))
        g[f2].append((e, f1))
        faces.add(f1)
        faces.add(f2)

    st = []
    parents = {}
    q = deque()

    cur = edges[0][0] if start is None else start
    seen = {cur}
    q.append(cur)
    parents[cur] = None

    if breadthfirst:
        # breadthfirst
        take = lambda: q.popleft()
        put = lambda x: q.append(x)
    else:
        # depthfirst
        take = lambda: q.pop()
        put = lambda x: q.append(x)

    while q:
        cur = take()
        for e, f in g[cur]:
            if f not in seen:
                st.append(canonical_key(cur, e, f))
                seen.add(f)
                put(f)
                parents[f] = (cur, e)

    assert len(set(st)) == len(st)  # unique
    assert set(st) <= set(edges)      # spanning tree is a subset of the graph

    return st, parents

def remove_if_present(name):
    o = bpy.data.objects
    if name in o:
        o.remove(o[name], do_unlink=True)

def object_from_bmesh(name, bm):
    ret = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    bm.to_mesh(ret.data)
    bpy.context.collection.objects.link(ret)
    return ret

def split_edges(name, mesh, spanning_tree):
    spanning_tree_edges = set(e for _, e, _ in spanning_tree)
    meshsplit = mesh.copy()
    remove_edges = [edge for edge in meshsplit.edges if edge.index not in spanning_tree_edges]
    bmesh.ops.split_edges(meshsplit, edges=remove_edges)

    return object_from_bmesh(name, meshsplit)

def origami(obj, breadthfirst=True):
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.faces.ensure_lookup_table()

    g = face_connectivity_graph(mesh)
    st, parents = spanning_tree(g, breadthfirst=breadthfirst)
    if len(parents) != len(mesh.faces):
        print('WARNING: spanning tree did not visit all faces, missing {}'.format(len(mesh.faces) - len(parents)))

    faces = {}
    for f_idx in parents:
        mesh_face = bmesh.new()
        f = mesh.faces[f_idx]
        mesh_face.faces.new([mesh_face.verts.new(x.co) for x in f.verts])
        faces[f_idx] = object_from_bmesh(f'{obj.name_full}face{f_idx:03d}', mesh_face)

    root = None
    for k, v in parents.items():
        if v is None:
            root = faces[k]
            continue
        parent, edge_idx = v
        o = faces[k]
        o.parent = faces[parent]

        orig_face = mesh.faces[k]
        parent_face = mesh.faces[parent]
        e = mesh.edges[edge_idx]
        midpoint = (e.verts[0].co + e.verts[1].co) / 2

        # setup the nex axis so Z is the face normal, Y points inwards along face, and X is ortho to both face normals
        # this makes it so that rotation in the positive X direction unfolds the face (makes it more like a flat sheet)
        dihedral = orig_face.normal.angle(parent_face.normal)
        if dihedral == 0:  # usually happens with flat sheets
            # i = e.verts[1].co - e.verts[0].co
            i = e.verts[1].co - e.verts[0].co
            # TODO figure out the right way to get the proper sign here, as the above is arb and is wrong
        else:
            i = orig_face.normal.cross(parent_face.normal)
        j = -i.cross(orig_face.normal)
        k = orig_face.normal

        new_origin = change_of_basis_matrix(midpoint, i, j, k)
        try:
            o.data.transform(new_origin.inverted())
            o.matrix_world = o.matrix_world @ new_origin
        except ValueError:
            print('WARNING matrix inversion failed')

        o['origami_original_angle'] = o.rotation_euler.x
        o['origami_dihedral_angle'] = dihedral

    assert root is not None

    return mesh, root, faces, st, parents

def dev():
    print('-' * 80)
    C = bpy.context
    D = bpy.data
    # obj = D.objects['Cube']
    # obj = D.objects['Icosphere']
    obj = D.objects['Plane']

    for o in list(D.objects.keys()):
        if o.startswith(f'{obj.name_full}face'):
            D.objects.remove(D.objects[o], do_unlink=True)

    mesh, root, faces, st, parents = origami(obj)
    for f in faces.values():
        f.show_axis = True

    # animate(root, 'ALL', 'UNFOLD', 0, 10, True)

def unfold_object(obj):
    angle = obj.get('origami_dihedral_angle')
    if angle is not None:
        obj.rotation_euler.x = obj.rotation_euler.x + angle

def fold_object(obj):
    angle = obj.get('origami_original_angle')
    if angle is not None:
        obj.rotation_euler.x = angle

def bfs(root):
    for child in root.children:
        yield from bfs(child)
    yield (root, )

def dfs(root):
    yield (root, )
    for child in root.children:
        yield from bfs(child)

def levels(root):
    def _levels(cur):
        yield cur.children
        for x in cur.children:
            yield from _levels(x)
    yield (root, )
    yield from _levels(root)

def tree_traversal(root, direction):
    if direction == 'BREADTH':
        return list(bfs(root))

    if direction == 'DEPTH':
        return list(dfs(root))

    if direction == 'LEVELS':
        return list(levels(root))

    if direction == 'ALL':
        return [list(itertools.chain.from_iterable(levels(root)))]

    raise KeyError(direction)

def animate(root, kind, direction, current_frame, frames_between, include_current=False):
    order = tree_traversal(root, kind)
    f = unfold_object if direction == 'UNFOLD' else fold_object
    if include_current:
        for obj in itertools.chain.from_iterable(order):
            obj.keyframe_insert(data_path='rotation_euler', frame=current_frame)
        current_frame += frames_between

    for objs in order:
        for obj in objs:
            f(obj)
            obj.keyframe_insert(data_path='rotation_euler', frame=current_frame)
        current_frame += frames_between

class OrigamiUnfold(bpy.types.Operator):
    bl_idname = 'object.origiamiunfold'
    bl_label = 'Origami Unfold'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.selected_objects

    def execute(self, context):
        for obj in context.selected_objects:
            unfold_object(obj)

        return {'FINISHED'}

class OrigamiFold(bpy.types.Operator):
    bl_idname = 'object.origiamifold'
    bl_label = 'Origami Fold'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.selected_objects

    def execute(self, context):
        for obj in context.selected_objects:
            fold_object(obj)

        return {'FINISHED'}

class Origamify(bpy.types.Operator):
    bl_idname = 'object.origamify'
    bl_label = 'Origamify'
    bl_options = {'REGISTER', 'UNDO'}

    breadthfirst: bpy.props.BoolProperty(name='Breadthfirst', default=True)
    # TODO: add an option for the starting face

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        obj = context.active_object
        origami(obj, breadthfirst=self.breadthfirst)

        return {'FINISHED'}

class OrigamiAnimate(bpy.types.Operator):
    bl_idname = 'object.origamianimate'
    bl_label = 'Origami Animate'
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(
        name='Animation Kind',
        items=[
            ('ALL', 'All Together', 'All Together'),
            ('BREADTH', 'Breadth First', 'Breadth First'),
            ('DEPTH', 'Depth First', 'Depth First'),
            ('LEVELS', 'Levels', 'Levels'),
        ],
    )
    direction: bpy.props.EnumProperty(
        name='Direction',
        items=[
            ('FOLD', 'Fold', 'Fold'),
            ('UNFOLD', 'Unfold', 'Unfold'),
        ],
    )
    frames_between: bpy.props.IntProperty(name='Frames Between', default=30, min=1)
    include_current: bpy.props.BoolProperty(name='Include Current Rotation', default=False)

    @classmethod
    def poll(cls, context):
        return context.selected_objects

    def execute(self, context):
        current_frame = context.scene.frame_current
        for obj in context.selected_objects:
            animate(obj, self.kind, self.direction, current_frame, self.frames_between, self.include_current)

        return {'FINISHED'}

classes = [
    Origamify,
    OrigamiUnfold,
    OrigamiFold,
    OrigamiAnimate,
]

def register():
    for klass in classes:
        bpy.utils.register_class(klass)

def unregister():
    for klass in classes:
        bpy.utils.unregister_class(klass)

if __name__ == '__dev__':
    # I have a script in a testing blendfile with the following two lines in it to run this script
    # filename = "/path/to/origami.py"
    # exec(compile(open(filename).read(), filename, 'exec'), {'__name__': '__dev__'})
    # dev()
    try:
        unregister()
    except Exception:
        pass
    register()

    dev()

elif __name__ == '__main__':
    try:
        unregister()
    except Exception:
        pass
    register()