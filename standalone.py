from collections import defaultdict, deque, namedtuple
from typing import List, Tuple, Dict

from functools import partial
from svgpathtools import svg2paths, Line
import shapely
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax
import jax.scipy.optimize

# TODO there is a big compute reduction we can do by only calculating the position
# of verts that are part of the cuts

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
    parents: Dict[int, Tuple[int, int]] # {fi: (parent, edge) | None}
    children: Dict[int, List[int]]
    segment_angles: Dict[int, float]  # degrees mapping from original segments
    face_angles: np.ndarray # (F) # radians, angle of a face

    # returns indices
    def verts_for_face(self, fi):
        offset, n = self.faces_idx[fi]
        return self.faces_packed[offset:offset+n]

@dataclass
class OptData:
    verts_idx: np.ndarray # (V', 2): (offset, n) into verts_packed
    verts_packed: np.ndarray # (V', 4) float
    mesh: Mesh
    mats: np.ndarray # (F, 4, 4) float
    correspondence: List[Tuple[int, int, int, int, int, int]]  # [(f1, f2, f1v1, f2v1, f1v2, f2v2)]

    # returns f4 coordinates
    def verts_for_face(self, fi):
        offset, n = self.verts_idx[fi]
        return self.verts_packed[offset:offset+n]

stroke_to_angle = {
    '#ff0000': 180,
    '#0000ff': -180,
    '#000000': 0,
    '#ffff00': 0,
}

angle_to_stroke = {
    180: '#ff0000',
    -180: '#0000ff',
    0: '#000000',
}

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
        for i2 in a - b, b - a:
            i = normalized(np.array([i2[0], i2[1], 0]))
            j = np.cross(i, k)
            cob = change_of_basis_matrix(mid, i, j, k)
            cob_inv = np.linalg.inv(cob)
            v2 = verts4[mesh.verts_for_face(fi)] @ cob_inv.T @ r4x(np.pi / 2).T
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
        if parent is None:
            root = fi
            continue
        parent_fi, ei = parent
        children[parent_fi].append(fi)
    return children

def prepare_opt_data(mesh, face_verts, mats):
    verts_idx = []
    verts_packed = []
    for vs in d_to_list(face_verts):
        verts_idx.append((len(verts_packed), len(vs)))
        verts_packed.extend(vs)
    verts_idx = np.array(verts_idx)
    verts_packed = np.array(verts_packed)

    correspondence = []
    for f1, ei, f2 in mesh.cuts:
        v0, v1 = mesh.edges[ei]
        correspondence.append((
            f1, f2,
            np.argmax(mesh.verts[mesh.verts_for_face(f1)] == v0),
            np.argmax(mesh.verts[mesh.verts_for_face(f2)] == v0),
            np.argmax(mesh.verts[mesh.verts_for_face(f1)] == v1),
            np.argmax(mesh.verts[mesh.verts_for_face(f2)] == v1),
            ))

    # do a pass down the tree that accumulates the inverse
    def go(i, mat):
        mats[i] = np.linalg.inv(mat) @ mats[i]
        mat = mat @ mats[i]
        for child in mesh.children[i]:
            go(child, mat)
    go(mesh.root, np.eye(4))

    mats = np.array(d_to_list(mats))

    return OptData(
        mesh=mesh,
        verts_idx=verts_idx,
        verts_packed=verts_packed,
        correspondence=correspondence,
        mats=mats,
    )

def apply_rotations(opt_data, angles, maxdepth=None):
    verts = {}
    def go(i, mat, depth=1):
        mat = mat @ opt_data.mats[i] @ r4x(angles[i])
        verts[i] = opt_data.verts_for_face(i) @ mat.T
        if maxdepth is not None and depth == maxdepth:
            return
        for child in opt_data.mesh.children[i]:
            go(child, mat, depth=depth+1)

    verts[opt_data.mesh.root] = opt_data.verts_for_face(opt_data.mesh.root)
    for child in opt_data.mesh.children[opt_data.mesh.root]:
        go(child, jnp.eye(4))

    return verts

#@jax.jit
#@jax.jit
def snap_opt_objective(x, opt_data):
    verts = apply_rotations(opt_data, x)
    loss = 0
    for f1, f2, f1v1, f2v1, f1v2, f2v2 in opt_data.correspondence:
        l1 = ((verts[f1][f1v1] - verts[f2][f2v1])**2).sum()
        l2 = ((verts[f1][f1v2] - verts[f2][f2v2])**2).sum()
        loss += l1 + l2

    return loss

def snap_opt(opt_data, x0=None):
    x0 = np.zeros(len(opt_data.verts), dtype=np.float32) if x0 is None else x0
    results = jax.scipy.optimize.minimize(
            snap_opt_objective,
            args=(opt_data,),
            x0=x0,
            method='BFGS',
            )
    return results

def get_angle(attr):
    if 'style' in attr:
        attr = dict(x.split(':') for x in attr['style'].split(';'))
    stroke = attr.get('stroke')
    if stroke is None:

        return 0
    angle = stroke_to_angle.get(stroke.lower())
    if angle is None:
        print('unknown stroke', attr)
        return 0
    return angle

def plot(lines, angles, polygons, st):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))

    for i, line in enumerate(lines):
        angle = angles.get(i, None)
        if angle is None:
            color = 'magenta'
        else:
            color = angle_to_stroke.get(angle, 'teal')
        x, y = line.xy
        axes[0].plot(x, y, color=color)
        # plt.plot(x, y)

    for poly in polygons.geoms:
        x, y = poly.buffer(-0.01).exterior.xy
        axes[1].plot(x, y)

    for f1, _, f2 in st:
        c1 = polygons.geoms[f1].centroid
        c2 = polygons.geoms[f2].centroid
        x = [c1.x, c2.x]
        y = [c1.y, c2.y]
        axes[1].plot(x, y, '.', color='grey')
        axes[1].plot(x, y, '--', color='grey')

    plt.tight_layout()
    plt.savefig('/tmp/plot.png')
    plt.close()

def plot3(faces, verts):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for fi, vs in verts.items():
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
    # center and make from -1 to 1
    points[:, 0] -= (xmax + xmin) / 2
    points[:, 1] -= (ymax + ymin) / 2
    xspan = (xmax - xmin) / 2
    yspan = (ymax - ymin) / 2
    scale = 1 / max(xspan, yspan)
    points *= scale

    lines = [shapely.LineString(ab) for ab in zip(points[0::2], points[1::2])]

    # this chops up lines at intersections
    segments = shapely.unary_union(lines, grid_size=0.001).geoms

    # now need to recover the correspondence between segments and the original line
    # so that we can get an angle per segment. we do this by buffering each line
    # and checking "which line covers each segment"
    tree = shapely.STRtree(segments)
    q = [line.buffer(0.001) for line in lines]

    segment_angles = {}
    for line_i, segment_i in tree.query(q, predicate='contains').T:
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

    # print('cuts', cuts)
    # print('dangles', dangles)
    # print('invalid', invalid)

    vert_index = {}  # {(x, y): i}
    faces = []  # [[vi0, vi1, ...]]
    edge_index = {}  # {(vi0, vi1): i}
    edge_faces = defaultdict(list)  # {ei: [fi0, fi1]}

    for f_i, polygon in enumerate(polygons.geoms):
        x, y = polygon.exterior.xy
        face_verts = []
        for xy in zip(x, y):
            if xy not in vert_index:
                vert_index[xy] = len(vert_index)
            face_verts.append(vert_index[xy])
        # face_verts has verts like [0, 1, 2, 3, 0]
        assert face_verts[0] == face_verts[-1]
        n = len(face_verts)
        for i in range(n - 1):
            e = canonical_key2(face_verts[i], face_verts[i + 1])
            if e not in edge_index:
                edge_index[e] = len(edge_index)
            ei = edge_index[e]
            edge_faces[ei].append(f_i)

        face_verts.pop() # remove duplicate vert at end
        faces.append(face_verts)

    verts = index_to_arr(vert_index, dtype=np.float32) # (V, 2)
    edges = index_to_arr(edge_index, dtype=int)  # (E, 2)

    # convert ragged face_verts [[vi0, vi1, vi2]] into
    faces_packed = []
    faces_idx = []
    for fs in faces:
        faces_idx.append((len(faces_packed), len(fs)))
        faces_packed.extend(fs)

    edge_faces_list = [] # [(f1, ei, f2)]
    for ei, fis in edge_faces.items():
        if len(fis) == 2:
            f1, f2 = fis
            edge_faces_list.append((f1, ei, f2))
        elif len(fis) > 2:
            print('wtf', ei, fis)

    st, parents = spanning_tree(edge_faces_list, start=central_poly)
    cuts = set(edge_faces_list) - set(st)

    face_angles = np.zeros(len(faces), dtype=np.float32)
    for fi, v in parents.items():
        if v is None:
            continue
        _, ei = v
        face_angles[fi] = segment_angles[ei]
    face_angles = np.radians(face_angles)

    return Mesh(
        root=central_poly,
        verts=verts,
        edges=edges,
        faces_idx=np.array(faces_idx),
        faces_packed=np.array(faces_packed),
        spanning_tree=st,
        parents=parents,
        children=parents_to_children(parents),
        cuts=cuts,
        segment_angles=segment_angles,
        face_angles=face_angles,
        )

# file = 'miura-ori.svg'
# file = 'test1.svg'
file = 'flasher1.svg'
# file = 'accordion.svg'
mesh = parse_svg(file)

# print(verts)
# print(faces)
# print(edges)

mats, face_verts = compute_matrices(mesh)
opt_data = prepare_opt_data(mesh, face_verts, mats)
# print(hash(opt_data))

# angles = np.ones(len(faces)) * np.radians(0)
# verts = apply_rotations(opt_data, angles)

x0 = mesh.face_angles * 0.0
import time
t0 = time.time()
results = snap_opt(opt_data, x0=x0)
t1 = time.time()
print(results)
angles = results.x
print('loss', results.fun)
print('took', t1 - t0)
verts = apply_rotations(opt_data, angles)
#verts = apply_rotations(opt_data, np.zeros(len(mesh.face_angles)))
# verts = {i: opt_data.verts_for_face(i) for i in mesh.parents}

#plot(segments, segment_angles, polygons, st)
#plot3(faces, verts)
plot3(None, verts)
