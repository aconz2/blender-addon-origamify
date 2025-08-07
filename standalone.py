from collections import defaultdict, deque, namedtuple
from typing import List, Tuple, Dict, NamedTuple, Callable
from dataclasses import dataclass
from itertools import chain
from functools import partial
import time

from svgpathtools import svg2paths, Line
import shapely

import numpy as np
import jax.numpy as jnp
import jax
import jax.scipy.optimize
import optax

# TODO there is a big compute reduction we can do by only calculating the position
# of verts that are part of the cuts
# and another for symmetry

stroke_to_angle = {
    '#ff0000': np.pi,  # red mountain
    '#0000ff': -np.pi, # blue valley
    '#000000': 0,
    '#ffff00': 0,  # yellow for trianglurization
}

angle_to_stroke = {
    1: '#ff0000',
    -1: '#0000ff',
    0: '#000000',
}

@dataclass
class Svg:
    lines: List[shapely.LineString]  # original lines
    line_angles: List[float]  # original line angles
    segments: List[shapely.LineString]  # chopped up segments
    segment_angles: Dict[int, float]  # degrees mapping from original segments
    polygons: List[shapely.Polygon]
    central_poly: int

# V' refers to the total number of duplicated vertices

@dataclass
class Mesh:
    root: int
    verts: np.ndarray # (V, 2) float
    edges: np.ndarray # (E, 2) int
    faces_idx: np.ndarray # (F, 2) int: (offset, n) into faces_packed
    faces_packed: np.ndarray # (V') int: index into verts
    spanning_tree: List[Tuple[int, int, int]]  # [(f1, ei, f2)]
    cuts: List[Tuple[int, int, int]]  # [(f1, ei, f2)]
    parents: Dict[int, Tuple[int, int] | None] # {fi: (parent, edge) | None}
    children: Dict[int, List[int]]
    face_angles: np.ndarray # (F) # radians, angle of a face
    correspondence: List[Tuple[int, int, int, int, int, int]]
    correspondence_arr: np.ndarray # (2, N) int verts_dup[arr[0]] == verts_dup[arr[1]]

    # returns indices
    def verts_for_face(self, fi):
        offset, n = self.faces_idx[fi]
        return self.faces_packed[offset:offset+n]

    def gen_f(self):
        gen = {}
        gen['r4x'] = r4x
        exec(gen_apply_rotations(mesh, name='apply_rotations'), gen)
        apply_rotations = jax.jit(gen['apply_rotations'])
        return OptF(
            apply_rotations=apply_rotations,
            )

class OptData(NamedTuple):
    # these verts are the duplicated vert positions for each face after being
    # inverse transformed so that apply_rotations with 0 rotation and the
    # mat from mats brings them all back into the original position
    verts_idx: np.ndarray # (V', 2): (offset, n) into verts_packed
    verts_packed: np.ndarray # (V', 4) float

    mats: np.ndarray # (F, 4, 4) float
    face_angles: np.ndarray # (F) radians,

    root: int
    correspondence_arr: np.ndarray # (2, N) int verts_dup[arr[0]] == verts_dup[arr[1]]

    # returns f4 coordinates
    def verts_for_face(self, fi):
        offset, n = self.verts_idx[fi]
        return self.verts_packed[offset:offset+n]

    def freeze(self):
        fields = ['mats', 'correspondence_arr', 'face_angles'] + [f'v{i}' for i in range(len(self.verts_idx))]
        T = namedtuple(f'OptDataFrozen{id(self)}', fields)
        kwargs = {
            'mats': self.mats,
            'correspondence_arr': self.correspondence_arr,
            'face_angles': self.face_angles,
        }
        for i in range(len(self.verts_idx)):
            kwargs[f'v{i}'] = self.verts_for_face(i)
        return T(**kwargs)


class OptF(NamedTuple):
    # OptData is actually OptDataFrozen here
    # returns all verts in a packed array
    # can index with mesh.faces_idx
    # angles -> (v, 3)
    # hom 4th component is chopped off
    apply_rotations: Callable[[np.ndarray, OptData], np.ndarray]

def d_to_arr(d, dtype=np.float32):
    arr = [None] * len(d)
    for k, v in d.items():
        arr[k] = v
    return np.array(arr, dtype=dtype)

def d_to_list(d):
    arr = [None] * len(d)
    for k, v in d.items():
        arr[k] = v
    return arr

def index_to_arr(d, dtype=np.float32):
    arr = [None] * len(d)
    for v, i in d.items():
        arr[i] = v
    return np.array(arr, dtype=dtype)

# for hom
def arrto4(arr, dtype=np.float32):
    n, d = arr.shape
    ret = np.zeros((n, 4), dtype=dtype)
    ret[:, :d] = arr
    ret[:, 3] = 1
    return ret

def mat3to4(mat, dtype=np.float32):
    ret = np.zeros((4, 4), dtype=dtype)
    ret[:3, :3] = mat
    ret[3, 3] = 1
    return ret

def t4(v, dtype=np.float32):
    z = v[2] if len(v) == 3 else 0
    return np.array([
        [1, 0, 0, v[0]],
        [0, 1, 0, v[1]],
        [0, 0, 1, z],
        [0, 0, 0, 1],
        ], dtype=dtype)

def r4x(angle, dtype=np.float32):
    cos = jnp.cos(angle)
    sin = jnp.sin(angle)
    return jnp.array([
        [1, 0, 0, 0],
        [0, cos, -sin, 0],
        [0, sin, cos, 0],
        [0, 0, 0, 1],
        ], dtype=dtype)

def normalized(v):
    return v / np.linalg.norm(v)

def change_of_basis_matrix(at, i, j, k):
    # assert np.isclose(1, np.linalg.norm(i))
    # assert np.isclose(1, np.linalg.norm(j))
    # assert np.isclose(1, np.linalg.norm(k))
    rot = mat3to4(np.array([i, j, k]))
    return t4(at) @ rot.T

def canonical_key2(a, b):
    if a < b:
        return a, b
    return b, a

def canonical_key3(face1, edge, face2):
    if face1 < face2:
        return (face1, edge, face2)
    return (face2, edge, face1)

def spanning_tree(edges, start=None):
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

    while q:
        cur = q.popleft()
        for e, f in g[cur]:
            if f not in seen:
                st.append(canonical_key3(cur, e, f))
                seen.add(f)
                q.append(f)
                parents[f] = (cur, e)

    assert len(set(st)) == len(st)  # unique
    assert set(st) <= set(edges)      # spanning tree is a subset of the graph

    return st, parents

def face_vert_not_on_edge(edge, face):
    for v in face:
        if v not in edge:
            return v
    raise ValueError('no such edge')

def compute_matrices(mesh):
    ret_face_verts = {}
    ret_mat = {}
    verts4 = arrto4(mesh.verts)
    for fi, parent in mesh.parents.items():
        if parent is None:
            ret_mat[fi] = np.eye(4, dtype=np.float32)
            ret_face_verts[fi] = verts4[mesh.verts_for_face(fi)]
            continue
        parent_fi, ei = parent
        a, b = mesh.verts[mesh.edges[ei]]
        mid = (a + b) / 2
        k = np.array([0, 0, 1])
        # to get the right edge direction for this edge, we check each direction and whether it moves the z coordinates of the face positive or negative
        # the other way I think would be to sort the verts CCW
        for i2 in a - b, b - a:
            i = normalized(np.array([i2[0], i2[1], 0]))
            j = np.cross(i, k)
            cob = change_of_basis_matrix(mid, i, j, k)
            cob_inv = np.linalg.inv(cob)
            v2 = verts4[mesh.verts_for_face(fi)] @ cob_inv.T @ r4x(np.pi / 2)
            zsum = v2[:, 2].sum()
            if zsum > 0:
                succ = True
                break
        assert succ
        ret_mat[fi] = cob
        ret_face_verts[fi] = verts4[mesh.verts_for_face(fi)] @ cob_inv.T

    return ret_mat, ret_face_verts

def parents_to_children(parents):
    children = defaultdict(list)
    root = None
    for fi, parent in parents.items():
        # even though we use defaultdict, b/c we later d_to_list this
        # we have to include an empty array for the children
        if fi not in children:
            children[fi] = []
        if parent is None:
            root = fi
            continue
        parent_fi, ei = parent
        children[parent_fi].append(fi)
    return children

def pack_ragged(arrs):
    idx = []
    packed = []
    for arr in arrs:
        idx.append((len(packed), len(arr)))
        packed.extend(arr)
    return np.array(idx), np.array(packed)

def prepare_opt_data(mesh, face_verts, mats):
    verts_idx, verts_packed = pack_ragged(d_to_list(face_verts))

    # do a pass down the tree that accumulates the inverse
    def go(i, mat):
        mats[i] = np.linalg.inv(mat) @ mats[i]
        mat = mat @ mats[i]
        for child in mesh.children[i]:
            go(child, mat)
    go(mesh.root, np.eye(4, dtype=np.float32))

    mats = np.array(d_to_list(mats))

    return OptData(
        verts_idx=verts_idx,
        verts_packed=verts_packed,
        mats=mats,
        root=mesh.root,
        correspondence_arr=mesh.correspondence_arr,
        face_angles=mesh.face_angles,
    )

# def apply_rotations(opt_data, angles, maxdepth=None):
#     verts = {}
#     def go(i, mat, depth=1):
#         mat = mat @ opt_data.mats[i] @ r4x(angles[i])
#         verts[i] = opt_data.verts_for_face(i) @ mat.T
#         if maxdepth is not None and depth == maxdepth:
#             return
#         for child in opt_data.children_for_face(i):
#             go(child, mat, depth=depth+1)
#
#     verts[opt_data.root] = opt_data.verts_for_face(opt_data.root)
#     for child in opt_data.children_for_face(opt_data.root):
#         go(child, jnp.eye(4))
#
#     return verts

def gen_apply_rotations(mesh, name='apply_rotations'):
    header = f'''
def {name}(angles, opt_data_frozen):
    import jax.numpy as jnp
    mats = opt_data_frozen.mats
'''
    footer = '''
    return ret
'''
    lines = []
    global_mat_num = 1
    def go(i, parent_mat_num):
        nonlocal global_mat_num
        my_mat_num = global_mat_num
        global_mat_num += 1
        lines.append(f'm{my_mat_num} = m{parent_mat_num} @ mats[{i}] @ r4x(angles[{i}])')
        lines.append(f'r{i} = opt_data_frozen.v{i} @ m{my_mat_num}.T')

        for child in mesh.children[i]:
            go(child, my_mat_num)

    lines.append(f'r{mesh.root} = opt_data_frozen.v{mesh.root}')
    lines.append(f'm0 = jnp.eye(4, dtype=jnp.float32)')
    for child in mesh.children[mesh.root]:
        go(child, 0)

    lines.append('ret = jnp.concatenate([')
    for i in sorted(mesh.parents):
        lines.append(f'    r{i}[:, :3],')
    lines.append('])')


    return header + '\n'.join(f'    {line}' for line in lines) + footer

def snap_loss(verts, opt_data_frozen):
    c = opt_data_frozen.correspondence_arr
    return ((verts[c[0]] - verts[c[1]]) ** 2).mean()

def snap_opt_objective(x, opt_data_frozen, apply_rotations):
    verts = apply_rotations(x, opt_data_frozen)
    return snap_loss(verts, opt_data_frozen)

def snap_opt(opt_data_frozen, opt_f, *, x0=None):
    x0 = np.zeros(len(opt_data.verts), dtype=np.float32) if x0 is None else x0
    results = jax.scipy.optimize.minimize(
            snap_opt_objective,
            args=(opt_data_frozen, opt_f.apply_rotations),
            x0=x0,
            method='BFGS',
            )
    return results

# tried jnp.clip(x, 0, 1) and it performed much worse
def to01(x):
    return (jnp.tanh(x) + 1) / 2

# params are -inf to inf, tanh -1 to 1, then move to [0, 1]
# and multiply by target
# this guarantees the angle has the same sign, and for 0 angled faces
# the angle will always be 0 (for triangulation)
def params_to_angles(x, target):
    return to01(x) * target

def snap_target_objective(x, opt_data_frozen, target, *, opt_f, vert_weight=1e4):
    x01 = to01(x)
    angles = x01 * target
    verts = opt_f.apply_rotations(angles, opt_data_frozen)
    la = snap_loss(verts, opt_data_frozen)

    # this is the average of the params values from [0, 1], where 1 means
    # we are at the target angle, so this loss is from [-1, 0] where -1 is
    # an exact match of the target
    lb = -x01.mean()

    loss = vert_weight * la + lb

    return loss, (loss, la, lb)

def snap_target(opt_data_frozen, opt_f, *, objective, target=None, x0=None, vert_mse=1e-6):
    target = opt_data.mesh.face_angles if target is None else target
    params = np.ones_like(target) if x0 is None else x0
    steps = 0
    solver = optax.adagrad(learning_rate=0.5)
    opt_state = solver.init(params)
    g = jax.jit(jax.grad(objective, has_aux=True))
    while True:
        grad, (loss, la, lb) = g(params, opt_data_frozen, target)
        # print('loss', loss)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        steps += 1
        if la < vert_mse:
            break
    print('steps', steps)
    loss, (_, la, lb) = objective(params, opt_data_frozen, target)
    angles = params_to_angles(params, target)
    return angles, params, loss, la, lb

def zsignarr(arr):
    return jnp.sign(arr) * (arr != 0)

@jax.jit
def jac_scale_objective(scale, jac, corr_arr, sign):
    y = (jac * (scale * sign)).sum(axis=-1)
    loss = ((y[corr_arr[0]] - y[corr_arr[1]]) ** 2).mean()
    return loss, loss

jac_scale_objective_grad = jax.jit(jax.grad(jac_scale_objective, has_aux=True))

def min_jac_scale_objective(jac, corr_arr, sign, mse=1e-6):
    steps = 0
    # solver = optax.adagrad(learning_rate=1.0)
    solver = optax.adam(learning_rate=1.0)
    params = np.ones(jac.shape[-1])
    opt_state = solver.init(params)
    while True:
        grad, loss = jac_scale_objective_grad(params, jac, corr_arr, sign)
        # print('loss', loss)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        steps += 1
        if loss < mse:
            break
    print('steps', steps)
    return params

def integrate(opt_data_frozen, opt_f, *, stepsize=0.01, steps=10, x0=None, maxangle=None):
    if x0 is None:
        x0 = np.zeros(len(opt_data_frozen.mats), dtype=np.float32)

    # jacfwd seems to be fastest
    apply_rotations_jac = jax.jit(jax.jacfwd(opt_f.apply_rotations))

    sign = zsignarr(opt_data_frozen.face_angles)
    corr_arr = opt_data_frozen.correspondence_arr

    cur = x0
    angles = [x0]

    for i in range(steps):
        print('step', i)
        jac = apply_rotations_jac(cur, opt_data_frozen)
        # jac has shape (verts, 3, angles)
        # for each coord of each vert, we get the partial deriv wrt each angle
        # we then seek a scale factor for each angle st when we scale those
        # derivs and sum them all (each angle's contributions), then we
        # minimize the difference of deriv between correspondence

        scale = min_jac_scale_objective(jac, corr_arr, sign)
        # print('scale', scale)
        cur += (scale / scale.max() * sign) * stepsize

        # print('cur', cur)
        print('known loss', snap_loss(opt_f.apply_rotations(cur, opt_data_frozen), opt_data_frozen))
        angles.append(cur)
        if maxangle is not None and cur.max() >= maxangle:
            break

    print(np.degrees(cur))
    print('done')
    return np.array(angles)

def get_angle(attr):
    if 'style' in attr:
        attr = dict(x.split(':') for x in attr['style'].split(';'))
    stroke = attr.get('stroke')
    if stroke is None:
        return 0

    angle = stroke_to_angle.get(stroke.lower())
    opacity = float(attr.get('stroke-opacity', '1'))
    if angle is None:
        print('unknown stroke', attr)
        return 0
    return opacity * angle

def zsign(x):
    if x == 0:
        return 0
    return np.sign(x)

def plot(mesh, svg):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(6, 18))

    for i, line in enumerate(svg.lines):
        x, y = line.xy
        c = line.centroid
        angle = svg.line_angles[i] / np.pi
        color = angle_to_stroke[zsign(angle)]
        axes[0].plot(x, y, color=color)
        axes[0].text(c.x, c.y, f'l{i}: {angle:.2f}', ha='center', va='center')

    for i, seg in enumerate(svg.segments):
        x, y = seg.xy
        c = seg.centroid
        angle = svg.segment_angles.get(i, None)
        if angle is None:
            color = 'magenta'
            axes[1].plot(x, y, color='magenta')
        else:
            color = angle_to_stroke[zsign(angle)]
            angle = angle / np.pi
            axes[1].plot(x, y, color=color)
            axes[1].text(c.x, c.y, f's{i}: {angle:.2f}', ha='center', va='center')

    for poly in svg.polygons.geoms:
        x, y = poly.buffer(-0.01).exterior.xy
        axes[2].plot(x, y)

    for f1, _, f2 in mesh.spanning_tree:
        c1 = svg.polygons.geoms[f1].centroid
        c2 = svg.polygons.geoms[f2].centroid
        x = [c1.x, c2.x]
        y = [c1.y, c2.y]
        axes[2].plot(x, y, '.', color='grey')
        axes[2].plot(x, y, '--', color='grey')

    for fi, parent in mesh.parents.items():
        if parent is None:
            continue
        _, ei = parent
        angle = mesh.face_angles[fi] / np.pi
        a, b = mesh.verts[mesh.edges[ei]]
        x = [a[0], b[0]]
        y = [a[1], b[1]]
        c = (a + b) / 2
        # axes[2].plot(x, y, '--', color='black')
        # axes[2].text(c[0] + 0.1, c[1], f'{angle:.0f}', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('/tmp/plot.png')
    plt.close()

def plot3(mesh, verts):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for offset, n in mesh.faces_idx:
        vs = verts[offset:offset+n]
        xs, ys, zs = list(vs[:, 0]), list(vs[:, 1]), list(vs[:, 2])
        xs.append(xs[0])
        ys.append(ys[0])
        zs.append(zs[0])
        ax.plot(xs, ys, zs)

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('/tmp/plot3.png')
    # plt.show()
    plt.close()

def plot_angles(angless):
    import matplotlib.pyplot as plt
    for i, a in enumerate(np.degrees(angless.T)):
        plt.plot(a, label=f'{i}')
    plt.tight_layout()
    plt.legend()
    plt.savefig('/tmp/plot_angles.png')

def total_vert_lengths(verts):
    s = 0
    for vs in verts.values():
        s += np.linalg.norm(vs[1:] - vs[:-1], axis=1).sum()
    return s

def parse_svg(filename):
    paths, attributes = svg2paths(filename)

    angles = []
    points = []
    for path, attr in zip(paths, attributes):
        angle = get_angle(attr)
        for part in path:
            if isinstance(part, Line):
                # flip y because svg +y goes down
                a = (part.start.real, -part.start.imag)
                b = (part.end.real, -part.end.imag)
                points.append(a)
                points.append(b)
                angles.append(angle)
            else:
                raise Exception('unhandled path part', type(part))

    points = np.array(points)
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    xspan = (xmax - xmin) / 2
    yspan = (ymax - ymin) / 2
    # center and make from -1 to 1
    points[:, 0] -= (xmax + xmin) / 2
    points[:, 1] -= (ymax + ymin) / 2
    scale = 1 / max(xspan, yspan)
    points *= scale

    lines = [shapely.LineString(ab) for ab in zip(points[0::2], points[1::2])]

    # this chops up lines at intersections
    segments = shapely.unary_union(lines, grid_size=0.001).geoms

    # now need to recover the correspondence between segments and the original line
    # so that we can get an angle per segment. we do this by buffering each line
    # and checking "which line covers each segment"
    tree = shapely.STRtree(segments)
    q = [line.buffer(0.01) for line in lines]

    segment_angles = {}
    for line_i, segment_i in tree.query(q, predicate='contains_properly').T:
        angle = angles[line_i]
        k = int(segment_i)
        if int(segment_i) in segment_angles:
            print(f'warn got double angle for {k}')
        segment_angles[k] = angle

    for i, segment in enumerate(segments):
        angle = segment_angles.get(i)
        if angle is None:
            print(f"MISSING angle for segment {i}")

    # turn lines into faces
    polygons, cuts, dangles, invalid = shapely.polygonize_full(segments)
    centroid_tree = shapely.STRtree([x.centroid for x in polygons.geoms])
    central_poly = int(centroid_tree.query_nearest(polygons.centroid, all_matches=False)[0])

    if len(cuts.geoms) != 0:
        raise ValueError('got cuts during polygonization', cuts)
    if len(dangles.geoms) != 0:
        raise ValueError('got dangles during polygonization', dangles)
    if len(invalid.geoms) != 0:
        raise ValueError('got invalid during polygonization', invalid)

    return Svg(
        lines=lines,
        line_angles=angles,
        segments=segments,
        segment_angles=segment_angles,
        polygons=polygons,
        central_poly=central_poly,
        )

def svg_to_mesh(svg):
    vert_index = {}  # {(x, y): i}
    edge_index = {}  # {(vi0, vi1): ei}
    segment_index = {} # {ei: si}
    for si, segment in enumerate(svg.segments):
        start, end = segment.coords
        if start not in vert_index:
            vert_index[start] = len(vert_index)
        if end not in vert_index:
            vert_index[end] = len(vert_index)
        e = canonical_key2(vert_index[start], vert_index[end])
        assert e not in edge_index
        ei = len(edge_index)
        edge_index[e] = ei
        segment_index[ei] = si

    faces = []  # [[vi0, vi1, ...]]
    edge_faces = defaultdict(list)  # {ei: [fi0, fi1]}

    for f_i, polygon in enumerate(svg.polygons.geoms):
        x, y = polygon.exterior.xy
        face_verts = []
        for xy in zip(x, y):
            face_verts.append(vert_index[xy])
        # face_verts has verts like [0, 1, 2, 3, 0]
        assert face_verts[0] == face_verts[-1]
        n = len(face_verts)
        for i in range(n - 1):
            ei = edge_index[canonical_key2(face_verts[i], face_verts[i+1])]
            edge_faces[ei].append(f_i)

        face_verts.pop() # remove duplicate vert at end
        faces.append(face_verts)

    verts = index_to_arr(vert_index, dtype=np.float32) # (V, 2)
    edges = index_to_arr(edge_index, dtype=int)  # (E, 2)

    # convert ragged face_verts [[vi0, vi1, vi2]]
    faces_idx, faces_packed = pack_ragged(faces)

    edge_faces_list = [] # [(f1, ei, f2)]
    for ei, fis in edge_faces.items():
        if len(fis) == 2:
            f1, f2 = fis
            edge_faces_list.append((f1, ei, f2))
        elif len(fis) > 2:
            print('wtf', ei, fis)

    st, parents = spanning_tree(edge_faces_list, start=svg.central_poly)
    cuts = set(edge_faces_list) - set(st)

    face_angles = np.zeros(len(faces), dtype=np.float32)
    for fi, v in parents.items():
        if v is None:
            continue
        _, ei = v
        segment = segment_index[ei]
        face_angles[fi] = svg.segment_angles[segment]

    correspondence = []
    for f1, ei, f2 in cuts:
        v0, v1 = edges[ei]
        correspondence.append((
            f1, f2,
            faces[f1].index(v0),
            faces[f2].index(v0),
            faces[f1].index(v1),
            faces[f2].index(v1),
            ))
    correspondence_arr = correspondence_to_arr(faces, correspondence)


    return Mesh(
        root=svg.central_poly,
        verts=verts,
        edges=edges,
        faces_idx=np.array(faces_idx),
        faces_packed=np.array(faces_packed),
        spanning_tree=st,
        parents=parents,
        children=parents_to_children(parents),
        cuts=cuts,
        face_angles=face_angles,
        correspondence=correspondence,
        correspondence_arr=correspondence_arr,
        )

def correspondence_to_arr(faces, correspondence):
    """faces: [[vi0, vi1, ...], ...]"""
    offset = np.array([len(arr) for arr in faces]).cumsum()
    offset[1:] = offset[:-1]
    offset[0] = 0
    l = []
    for f1, f2, f1v1, f2v1, f1v2, f2v2 in correspondence:
        l.append((f1v1 + offset[f1], f2v1 + offset[f2]))
        l.append((f1v2 + offset[f1], f2v2 + offset[f2]))
    return np.array(l).T

def optimize_mesh(mesh):
    face_with_cuts = set()
    verts_for_face = defaultdict(list)
    cut_verts = 0
    for f1, f2, f1v1, f2v1, f1v2, f2v2 in mesh.correspondence:
        face_with_cuts.add(f1)
        face_with_cuts.add(f2)
        verts_for_face[f1].append(f1v1)
        verts_for_face[f2].append(f2v1)
        verts_for_face[f1].append(f1v2)
        verts_for_face[f2].append(f2v2)
        cut_verts += 4

    print(face_with_cuts)
    print(verts_for_face)
    print(cut_verts)
    print(len(mesh.verts))
    has_cut = set()
    def go(i):
        has_cuts = i in face_with_cuts
        for child in mesh.children[i]:
            has_cuts |= go(child)
        if has_cuts:
            has_cut.add(i)
        return has_cuts
    go(mesh.root)
    print(has_cut)

# file = 'miura-ori.svg'
# file = 'test1.svg'
#file = 'flasher3.svg'
file = 'flasher5.svg'
# file = 'accordion.svg'
svg = parse_svg(file)
mesh = svg_to_mesh(svg)

# print(len(mesh.faces_idx))
# optimize(mesh)
# import sys
# sys.exit(0)

# plot(mesh, svg)

print(gen_apply_rotations(mesh))
# print(gen_snap_opt_objective(mesh))


opt_f = mesh.gen_f()

mats, face_verts = compute_matrices(mesh)
opt_data = prepare_opt_data(mesh, face_verts, mats)
opt_data_frozen = opt_data.freeze()

target = mesh.face_angles * 0.8

# method = 'snap'
# method = 'target'
method = 'integrate'

if method == 'integrate':
    t0 = time.time()
    # angless = integrate(opt_data_frozen, opt_f, steps=400, maxangle=np.pi * 0.95)
    angless = integrate(opt_data_frozen, opt_f, steps=500)
    t1 = time.time()
    plot_angles(angless)
    angles = angless[-1]

elif method == 'snap':
    t0 = time.time()
    results = snap_opt(opt_data_frozen, opt_f, x0=target)
    t1 = time.time()
    angles = results.x
    print('loss', results.fun)
    print('nfev', results.nfev)
    print('target', np.degrees(target))
    print('angles', np.degrees(angles))
    print('diff', np.degrees(target - angles))
    print('angle mse', ((target-angles)**2).mean())

elif method == 'target':
    t0 = time.time()
    # the more unsure we are about the fold angles from the SVG, the higher vert_weight should be
    snap_target_objective_ = jax.jit(partial(snap_target_objective, opt_f=opt_f, vert_weight=1e4))
    angles, params, loss, la, lb = snap_target(
            opt_data_frozen,
            opt_f,
            target=target,
            objective=snap_target_objective_,
            )
    t1 = time.time()
    print('params', params)
    # print('st', mesh.spanning_tree)
    # print('segments', mesh.segment_angles)
    print('la', la, 'lb', lb)
    print('loss', loss)
    print('target', np.degrees(target))
    print('angles', np.degrees(angles))
    print('diff', np.degrees(target - angles))
    print('angle mse', ((target-angles)**2).mean())

print('took', t1 - t0)
verts = opt_f.apply_rotations(angles, opt_data_frozen)
print('vert mse', snap_loss(verts, opt_data_frozen))

plot3(mesh, verts)

# subdivide
if False:
    final_angles = angles
    final_params = params

    snap_target_objective_ = jax.jit(partial(snap_target_objective, opt_f=opt_f, vert_weight=1e4))
    n = 10
    for t in np.linspace(0, 1, n+2)[1:-1]:
        target = final_angles * t
        x0 = final_params * t
        t0 = time.time()
        angles, params, loss, la, lb = snap_target(
                opt_data_frozen,
                opt_f,
                x0=x0,
                target=target,
                objective=snap_target_objective_,
                )
        t1 = time.time()
        print('took', t1 - t0)
        print('la', la, 'lb', lb)
        print('angle mse', ((target-angles)**2).mean())
        verts = opt_f.apply_rotations(angles, opt_data_frozen)
        print('vert mse', snap_loss(verts, opt_data_frozen))
