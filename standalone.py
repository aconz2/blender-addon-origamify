from svgpathtools import svg2paths, Line
import shapely
from collections import defaultdict, deque, namedtuple
import networkx as nx
import numpy as np

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

# def d_to_arr(d):
#     arr = [None] * len(d)
#     for k, v in d.items():
#         arr[k] = v
#     return np.array(arr, dtype=np.float32)
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

def mat3to4(mat):
    ret = np.zeros((4, 4))
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
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, cos, -sin, 0],
        [0, sin, cos, 0],
        [0, 0, 0, 1],
        ], dtype=dtype)

def r4z(angle, dtype=float):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([
        [cos, -sin, 0, 0],
        [sin, cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        ], dtype=dtype)

def normalized(v):
    return v / np.linalg.norm(v)

def rotate_about_edge(v, rx):
    rz = np.atan2(v[1], v[0])
    return t4(v) @ r4z(-rz) @ r4x(rx) @ r4z(rz) @ t4(-v)
    # return t4(-v) @ r4z(-rz) @ r4x(rx) @ r4z(rz) @ t4(v)
    # return t4(v) @ r4z(rz) @ r4x(rx) @ r4z(-rz) @ t4(-v)

def change_of_basis_matrix(at, i, j, k):
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

# these are the correcly pointing edge vectors that when rotated about positively, the face moves into +Z
def compute_matrices(parents, verts, edges, faces):
    ret_face_verts = {}
    ret_mat = {}
    verts4 = arrto4(verts)
    for fi, parent in parents.items():
        if parent is None:
            ret_face_verts[fi] = verts4[faces[fi]]
            continue
        parent_fi, ei = parent
        a, b = verts[edges[ei]]
        face_verts = verts4[faces[fi]]
        ev = a - b
        mid = (a + b) / 2
        v2 = face_verts @ rotate_about_edge(ev, np.pi / 2).T
        # the verts on the edge will stay at 0, but any other verts will get moved up or down
        zsum = v2[:, 2].sum()
        if zsum < 0:
            ev = b - a
        i = normalized(np.array([ev[0], ev[1], 0]))
        k = np.array([0, 0, 1])
        j = np.cross(i, k)
        cob = change_of_basis_matrix(mid, i, j, k)
        cob_inv = np.linalg.inv(cob)
        ret_mat[fi] = cob
        ret_face_verts[fi] = face_verts @ cob_inv.T

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
            faces[f1].index(v1),
            faces[f2].index(v0),
            faces[f2].index(v1),
            ))

    return OptData(
        verts = verts,
        faces = faces,
        children = children,
        correspondence = correspondence,
        root = root,
        mats = mats,
    )

def apply_rotations(opt_data, angles):
    verts = {}
    def go(i, mat, depth=0):
        mat = mat @ opt_data.mats[i] @ r4x(angles[i])
        verts[i] = opt_data.verts[i] @ mat.T
        if depth == 0:
            return
        for child in opt_data.children[i]:
            go(child, mat, depth=depth+1)
            # break

    verts[opt_data.root] = opt_data.verts[opt_data.root]
    for child in opt_data.children[opt_data.root]:
        go(child, np.eye(4, dtype=np.float32))
    return verts

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
        x, y = poly.buffer(-1).exterior.xy
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

    plt.tight_layout()
    plt.savefig('/tmp/plot3.png')
    plt.close()

# file = 'miura-ori.svg'
# file = 'test1.svg'
file = 'flasher1.svg'
paths, attributes = svg2paths(file)

lines = []
angles = []
for path, attr in zip(paths, attributes):
    angle = get_angle(attr)
    for part in path:
        if isinstance(part, Line):
            # flip y because svg +y goes down
            a = (part.start.real, -part.start.imag)
            b = (part.end.real, -part.end.imag)
            lines.append(shapely.LineString([a, b]))
            angles.append(angle)
        else:
            raise Exception('unhandled path part', type(part))

segments = shapely.unary_union(lines, grid_size=0.1).geoms
tree = shapely.STRtree(segments)

segment_angles = {}

q = [line.buffer(0.1) for line in lines]
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

# angles = np.ones(len(faces)) * np.radians(10)
angles = np.ones(len(faces)) * np.radians(90)
verts = apply_rotations(opt_data, angles)

plot(segments, segment_angles, polygons, st)
plot3(faces, verts)
