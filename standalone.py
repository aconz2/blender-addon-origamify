from collections import defaultdict, deque, namedtuple

from functools import partial
from svgpathtools import svg2paths, Line
import shapely

import numpy as np
import jax.numpy as jnp
import jax
import jax.scipy.optimize

# TODO there is a big compute reduction we can do by only calculating the position
# of verts that are part of the cuts

OptData = namedtuple('OptData', ['verts', 'faces', 'children', 'correspondence', 'root', 'mats'])

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

def d_to_arr(d):
    arr = [None] * len(d)
    for k, v in d.items():
        arr[k] = v
    return np.array(arr, dtype=np.float32)

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
def arrto4(arr):
    n, d = arr.shape
    ret = np.zeros((n, 4))
    ret[:, :d] = arr
    ret[:, 3] = 1
    return ret

def mat3to4(mat, dtype=float):
    ret = np.zeros((4, 4), dtype=dtype)
    ret[:3, :3] = mat
    ret[3, 3] = 1
    return ret

def t4(v, dtype=float):
    z = v[2] if len(v) == 3 else 0
    return np.array([
        [1, 0, 0, v[0]],
        [0, 1, 0, v[1]],
        [0, 0, 1, z],
        [0, 0, 0, 1],
        ], dtype=dtype)

def r4x(angle, dtype=float):
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

# def project(a, b):
#     return np.dot(a, normalized(b))
#
# def vector_rejection(a, b):
#     return a - project(a, b)

def compute_matrices(parents, verts, edges, faces):
    ret_face_verts = {}
    ret_mat = {}
    verts4 = arrto4(verts)
    for fi, parent in parents.items():
        if parent is None:
            ret_mat[fi] = np.eye(4, dtype=float)
            ret_face_verts[fi] = verts4[faces[fi]]
            continue
        parent_fi, ei = parent
        a, b = verts[edges[ei]]
        mid = (a + b) / 2
        k = np.array([0, 0, 1])
        for i2 in a - b, b - a:
            i = normalized(np.array([i2[0], i2[1], 0]))
            j = np.cross(i, k)
            cob = change_of_basis_matrix(mid, i, j, k)
            cob_inv = np.linalg.inv(cob)
            v2 = verts4[faces[fi]] @ cob_inv.T @ r4x(np.pi / 2).T
            zsum = v2[:, 2].sum()
            if zsum > 0:
                succ = True
                break
        assert succ
        ret_mat[fi] = cob
        ret_face_verts[fi] = verts4[faces[fi]] @ cob_inv.T

        # c = verts[face_vert_not_on_edge(edges[ei], faces[fi])]
        # j = normalized(vector_rejection(c - a, a - b))
        # mid = (a + b) / 2
        # j = np.array([j[0], j[1], 0])
        # k = np.array([0, 0, 1])
        # i = np.cross(k, j)
        # cob = change_of_basis_matrix(mid, i, j, k)
        # cob_inv = np.linalg.inv(cob)
        # ret_mat[fi] = cob
        # ret_face_verts[fi] = verts4[faces[fi]] @ cob_inv.T

    return ret_mat, ret_face_verts

def prepare_opt_data(parents, cuts, verts, edges, faces, edge_vert_i, mats):
    children = defaultdict(list)
    root = None
    for fi, parent in parents.items():
        if parent is None:
            root = fi
            continue
        parent_fi, ei = parent
        children[parent_fi].append(fi)

    assert root is not None

    correspondence = []
    for f1, ei, f2 in cuts:
        v0, v1 = edge_vert_i[ei]
        correspondence.append((
            f1, f2,
            faces[f1].index(v0),
            faces[f2].index(v0),
            faces[f1].index(v1),
            faces[f2].index(v1),
            ))

    # do a pass down the tree that accumulates the inverse
    def go(i, mat):
        mats[i] = np.linalg.inv(mat) @ mats[i]
        mat = mat @ mats[i]
        for child in children[i]:
            go(child, mat)
    go(root, np.eye(4))

    return OptData(
        verts = d_to_list(verts),
        faces = faces,
        children = children,
        correspondence = correspondence,
        root = root,
        mats = mats,
    )

def apply_rotations(opt_data, angles, maxdepth=None):
    verts = {}
    def go(i, mat, depth=1):
        mat = mat @ opt_data.mats[i] @ r4x(angles[i])
        verts[i] = opt_data.verts[i] @ mat.T
        if maxdepth is not None and depth == maxdepth:
            return
        for child in opt_data.children[i]:
            go(child, mat, depth=depth+1)

    verts[opt_data.root] = opt_data.verts[opt_data.root]
    for child in opt_data.children[opt_data.root]:
        go(child, jnp.eye(4))

    return verts

@jax.jit
def snap_opt_objective(x, opt_data):
    verts = apply_rotations(opt_data, x)
    loss = 0
    for f1, f2, f1v1, f2v1, f1v2, f2v2 in opt_data.correspondence:
        l1 = ((verts[f1][f1v1] - verts[f2][f2v1])**2).sum()
        l2 = ((verts[f1][f1v2] - verts[f2][f2v2])**2).sum()
        loss += l1 + l2

    return loss

def snap_opt(opt_data, x0=None):
    x0 = np.zeros(len(opt_data.verts)) if x0 is None else x0
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

# file = 'miura-ori.svg'
# file = 'test1.svg'
file = 'flasher1.svg'
# file = 'accordion.svg'
paths, attributes = svg2paths(file)

lines = []
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
points[:, 0] -= (xmax + xmin) / 2
points[:, 1] -= (ymax + ymin) / 2

# center and make from -1 to 1
xspan = (xmax - xmin) / 2
yspan = (ymax - ymin) / 2
xmin, xmax = points[:, 0].min(), points[:, 0].max()
ymin, ymax = points[:, 1].min(), points[:, 1].max()
scale = 1 / max(xspan, yspan)
points *= scale

for a, b in zip(points[0::2], points[1::2]):
    lines.append(shapely.LineString([a, b]))

segments = shapely.unary_union(lines, grid_size=0.001).geoms
tree = shapely.STRtree(segments)

segment_angles = {}

q = [line.buffer(0.001) for line in lines]
for line_i, segment_i in tree.query(q, predicate='contains').T:
    # print(line_i, segment_i)
    angle = angles[line_i]
    if int(segment_i) in segment_angles:
        print('warn got double angle for {}'.format(int(segment_i)))
    segment_angles[int(segment_i)] = angle
    # print(f'sgement {segment_i} should have angle {angle}')

for i, segment in enumerate(segments):
    # print(segment)
    angle = segment_angles.get(i)
    if angle is None:
        print(f"MISSING angle for segment {i}")

polygons, cuts, dangles, invalid = shapely.polygonize_full(segments)
centroid_tree = shapely.STRtree([x.centroid for x in polygons.geoms])
central_poly = int(centroid_tree.query_nearest(polygons.centroid, all_matches=False)[0])

print('cuts', cuts)
print('dangles', dangles)
print('invalid', invalid)

vert_index = {}  # {(x, y): i}
faces = []  # [[vi0, vi1, ...]]
edge_index = {}  # {(vi0, vi1): i}
edge_vert_i = {} # {i: (vi0, vi1)}
edge_faces = defaultdict(list)  # {ei: [fi0, fi1]}

for f_i, polygon in enumerate(polygons.geoms):
    x, y = polygon.exterior.xy
    face_verts = []
    for xy in zip(x, y):
        # print(xy)
        if xy not in vert_index:
            vert_index[xy] = len(vert_index)
        face_verts.append(vert_index[xy])
    # face_verts has verts like [0, 1, 2, 3, 0]
    assert face_verts[0] == face_verts[-1]
    n = len(face_verts)
    for i in range(n - 1):
        j = i + 1
        e = canonical_key2(face_verts[i], face_verts[j])
        if e not in edge_index:
            edge_vert_i[len(edge_index)] = e
            edge_index[e] = len(edge_index)
        ei = edge_index[e]
        edge_faces[ei].append(f_i)

    face_verts.pop() # remove duplicate vert at end
    faces.append(face_verts)

verts = index_to_arr(vert_index, dtype=np.float32) # (V, 2)
edges = index_to_arr(edge_index, dtype=int)  # (E, 2)

# print(verts)
# print(faces)
# print(edges)

edge_faces_list = [] # [(f1, ei, f2)]
for ei, fis in edge_faces.items():
    if len(fis) == 2:
        f1, f2 = fis
        edge_faces_list.append((f1, ei, f2))
    elif len(fis) > 2:
        print('wtf', ei, fis)

print(edge_faces_list)
st, parents = spanning_tree(edge_faces_list, start=central_poly)
cuts = set(edge_faces_list) - set(st)
print('st', st)
print('parents', parents)
print('cuts', cuts)
mats, face_verts = compute_matrices(parents, verts, edges, faces)
opt_data = prepare_opt_data(parents, cuts, face_verts, edges, faces, edge_vert_i, mats)

# angles = np.ones(len(faces)) * np.radians(0)
# verts = apply_rotations(opt_data, angles)

x0 = np.radians(d_to_arr(segment_angles)) * 0.5
import time
t0 = time.time()
results = snap_opt(opt_data, x0=x0)
t1 = time.time()
angles = results.x
print('loss', results.fun)
print('took', t1 - t0)
verts = apply_rotations(opt_data, angles)

plot(segments, segment_angles, polygons, st)
plot3(faces, verts)
