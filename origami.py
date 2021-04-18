
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

TOL = 1e-6

class SpanningTreeMissingFaces(Exception):
    def __init__(self, n_missing):
        self.n_missing = n_missing
        super().__init__()

def is_0_180(x, tol=TOL):
    return math.isclose(x, 0, rel_tol=tol) or math.isclose(x, math.pi, rel_tol=tol)

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

def face_connectivity_graph(obj, mesh, use_seams=False):
    """
    Returns a list of triples (face_idx0, edge_idx, face_idx1) which correspond to the linkage information
    face_idx0 < face_idx1 to dedup
    raises Exception if edge connects more than two faces
    """
    ret = []

    for edge in mesh.edges:
        if use_seams and obj.data.edges[edge.index].use_seam or len(edge.link_faces) != 2:
            continue
        ret.append(canonical_key(edge.link_faces[0].index, edge.index, edge.link_faces[1].index))

    return ret

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
        take = q.popleft
        put = q.append
    else:
        # depthfirst
        take = q.pop
        put = q.append

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

def face_vert_not_on_edge(face, edge):
    # return next(iter(set(face.verts) - set(edge.verts)))
    edge_verts = set(edge.verts)
    for v in face.verts:
        if v not in edge_verts:
            return v

    assert False

def vector_rejection(a, b):
    return a - a.project(b)

def origami(obj, breadthfirst=True, use_seams=False):
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.edges.ensure_lookup_table()
    mesh.faces.ensure_lookup_table()

    g = face_connectivity_graph(obj, mesh, use_seams=use_seams)
    start = mesh.faces.active.index if mesh.faces.active else None
    st, parents = spanning_tree(g, breadthfirst=breadthfirst, start=start)
    if len(parents) != len(mesh.faces):
        raise SpanningTreeMissingFaces(len(mesh.faces) - len(parents))
        print('WARNING: spanning tree did not visit all faces, missing {}'.format(len(mesh.faces) - len(parents)))

    faces = {}
    for f_idx in parents:
        mesh_face = bmesh.new()
        f = mesh.faces[f_idx]
        mesh_face.faces.new([mesh_face.verts.new(x.co) for x in f.verts])
        bmesh.ops.recalc_face_normals(mesh_face, faces=mesh_face.faces)
        faces[f_idx] = object_from_bmesh(f'{obj.name_full}face{f_idx:03d}', mesh_face)

    root = None
    for k, v in parents.items():
        if v is None:
            root = faces[k]
            continue
        parent, edge_idx = v
        o = faces[k]
        o.parent = faces[parent]

        orig_face   = mesh.faces[k]
        parent_face = mesh.faces[parent]

        e = mesh.edges[edge_idx]
        midpoint = (e.verts[0].co + e.verts[1].co) / 2
        ev = e.verts[1].co - e.verts[0].co

        # setup the new axis so Z is the face normal, Y points inwards along face, and X is ortho to both face normals
        # this makes it so that rotation in the positive X direction rotates +Z according to the right hand rule (counter clockwise)

        # j is a vector along the face, perpindicular to the hinge edge e
        j = vector_rejection(face_vert_not_on_edge(orig_face, e).co - e.verts[0].co, ev)
        k = orig_face.normal
        i = j.cross(k)

        dihedral = orig_face.normal.angle(parent_face.normal)
        assert 0 <= dihedral <= math.pi

        # Y axis along the parent
        j_parent = vector_rejection(face_vert_not_on_edge(parent_face, e).co - e.verts[0].co, ev)

        # because face normals set the sign of rotation and we can have alternate normals between the two faces
        # we check each sign and each rotation for the proper one
        # the proper one is the one where rotating by it forms a straight line between the two j (Y) vectors
        angles = [
            dihedral,
            -dihedral,
            math.pi - dihedral,
            -(math.pi - dihedral),
        ]

        dihedral_flatten_delta = None
        for angle in angles:
            jj = rotate_about_axis(i, angle) @ j
            if math.isclose(j_parent.angle(jj), math.pi, rel_tol=TOL):
                dihedral_flatten_delta = angle
                new_dihedral = (rotate_about_axis(i, angle) @ k).angle(parent_face.normal)
                assert is_0_180(new_dihedral)
                break

        assert dihedral_flatten_delta is not None

        new_origin = change_of_basis_matrix(midpoint, i, j, k)
        try:
            o.data.transform(new_origin.inverted())
            o.matrix_world = o.matrix_world @ new_origin
        except ValueError:
            print('WARNING matrix inversion failed')

        o['origami_original_angle'] = o.rotation_euler.x
        o['origami_dihedral_angle'] = dihedral  # not used anymore but could still be useful
        o['origami_unfold_angle'] = o.rotation_euler.x + dihedral_flatten_delta

    assert root is not None

    # fixup normals, still not sure why they are sometimes flipped
    # BUG this isn't reliable, for the most part, the noraml is always inverted and needs flipping, but sometimes there is a false negative or two
    for f_idx in parents:
        face = mesh.faces[f_idx]
        obj = faces[f_idx]

        m = bmesh.new()
        m.from_mesh(obj.data)
        m.transform(obj.matrix_world)
        bmesh.ops.recalc_face_normals(m, faces=m.faces)
        m.faces.ensure_lookup_table()
        assert len(m.faces) == 1
        obj_face = m.faces[0]
        if not math.isclose(obj_face.normal.angle(face.normal), 0):
            # we start with a fresh mesh, or we could invert the obj.matrix_world transform
            m = bmesh.new()
            m.from_mesh(obj.data)
            for f in m.faces:
                f.normal_flip()
            m.to_mesh(obj.data)

    return mesh, root, faces, st, parents

def dev():
    print('-' * 80)
    C = bpy.context
    D = bpy.data

    for o in list(D.objects.keys()):
        if 'face' in o:
            D.objects.remove(D.objects[o], do_unlink=True)

    for name in ['Cube', 'Tetrahedron', 'AccordionCrinkled', 'AccordionCrinkledAlternating', 'Plane']:
        obj = D.objects[name]
        mesh, root, faces, st, parents = origami(obj)
        for f in faces.values():
            f.show_axis = True
        unfold_object(root, recursively=True)
        obj.hide_viewport = True

    # animate(root, 'ALL', 'UNFOLD', 0, 10, True)

def unfold_object(obj, recursively=False, levels=-2):
    if levels == -1:
        return
    angle = obj.get('origami_unfold_angle')
    if angle is not None:
        obj.rotation_euler.x = angle

    if recursively:
        for child in obj.children:
            unfold_object(child, recursively, levels - 1)

def fold_object(obj, recursively=False):
    angle = obj.get('origami_original_angle')
    if angle is not None:
        obj.rotation_euler.x = angle

    if recursively:
        for child in obj.children:
            fold_object(child, recursively)

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
    """UnFold an origami object flat"""
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
    """Fold an origami object back to its original state"""
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
    """Create nested object from faces for unfolding"""
    bl_idname = 'object.origamify'
    bl_label = 'Origamify'
    bl_options = {'REGISTER', 'UNDO'}

    breadthfirst: bpy.props.BoolProperty(name='Breadthfirst', default=True)
    constrain_root: bpy.props.BoolProperty(name='Constrain Root', default=False, description='Add an X Limit Rotation Constraint to the root object to always be 0')
    unfold: bpy.props.BoolProperty(name='Unfold', default=False)
    use_seams: bpy.props.BoolProperty(name='Use Seams', default=False, description='Respect marked seams by never hinging on them')

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        obj = context.active_object
        try:
            mesh, root, faces, st, parents = origami(obj, breadthfirst=self.breadthfirst, use_seams=self.use_seams)
        except SpanningTreeMissingFaces as e:
            self.report({'ERROR'}, f'Spanning tree did not cover whole object, missing {e.n_missing} faces. Maybe you have too many seams')
            return {'FINISHED'}

        if self.constrain_root:
            root.constraints.new(type='LIMIT_ROTATION')
            root.constraints['Limit Rotation'].use_limit_x = True
            root.constraints['Limit Rotation'].min_x = 0
            root.constraints['Limit Rotation'].max_x = 0

        if self.unfold:
            unfold_object(root, recursively=True)

        return {'FINISHED'}

class OrigamiAnimate(bpy.types.Operator):
    """Animate the folding or unfolding of an origami object"""
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

class OrigamiMenu(bpy.types.Menu):
    bl_label = 'Origamify'
    bl_idname = 'OBJECT_MT_origamify'

    def draw(self, context):
        for klass in classes:
            self.layout.operator(klass.bl_idname)


def menu_func(self, context):
    self.layout.menu(OrigamiMenu.bl_idname)

def register():
    bpy.utils.register_class(OrigamiMenu)
    for klass in classes:
        bpy.utils.register_class(klass)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(OrigamiMenu)
    for klass in classes:
        bpy.utils.unregister_class(klass)
    bpy.types.VIEW3D_MT_object.remove(menu_func)

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
    register()
