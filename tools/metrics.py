import numpy as np
from tools.PCLosses import acc_comp
from shapely.geometry import Polygon
import torch
from pyquaternion import Quaternion

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def estimateAccuracy(box_a, box_b, dim=3) -> float:
    if dim == 3:
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:
        return np.linalg.norm(
            box_a.center[[0, 2]] - box_b.center[[0, 2]], ord=2)

def estimateAccuracywithboxes3d(box_a, box_b, dim=3) -> float:
    box_a_xyz = box_a[0:3]
    box_a_hwl = box_a[3:6]
    box_a_ry = box_a[-1]

    center_a = torch.tensor([box_a_xyz[0], box_a_xyz[1] - box_a_hwl[0] / 2, box_a_xyz[2]]).cuda().float()

    box_b_xyz = box_b[0:3]
    box_b_hwl = box_b[3:6]
    box_b_ry = box_b[-1]

    center_b = torch.tensor([box_b_xyz[0], box_b_xyz[1] - box_b_hwl[0] / 2, box_b_xyz[2]]).cuda().float()


    if dim == 3:
        return float(torch.norm(center_a - center_b))
    elif dim == 2:
        return float(torch.norm(center_a[0, 2] - center_b[0, 2]))


def fromBoxToPoly(box):
    return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))


def corners(center, rois_tmp_hwl, rotation_matrix, wlh_factor: float = 1.0):
    """"
    Returns the bounding box corners.
    :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    rois_tmp_hwl = rois_tmp_hwl * wlh_factor
    h, w, l = rois_tmp_hwl
    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    corners = torch.tensor(np.array([l.cpu().numpy() / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1]),
                                     w.cpu().numpy() / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1]),
                                     h.cpu().numpy() / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])])).cuda().float()
    # Rotate
    corners = torch.mm(rotation_matrix, corners)
    # Translate
    corners = center.float() + corners.float()
    return corners


def estimateOverlapwithboxes3d(box_a, box_b, dim=2) -> float:
    if torch.equal(box_a, box_b):
        return 1.0

    rois_tmp_xyz_a = box_a[0:3]
    rois_tmp_hwl_a = box_a[3:6]
    rois_tmp_ry_a = box_a[-1]
    h_a, w_a, l_a = rois_tmp_hwl_a
    center_a = torch.tensor([rois_tmp_xyz_a[0], rois_tmp_xyz_a[1] - rois_tmp_hwl_a[0] / 2, rois_tmp_xyz_a[2]]).reshape((3, 1)).cuda().float()
    rotation_matrix_a = torch.tensor((Quaternion(axis=[0, 1, 0], radians=float(rois_tmp_ry_a)) *
                                   Quaternion(axis=[1, 0, 0], radians=np.pi / 2)).rotation_matrix).cuda().float()
    corners_a = corners(center_a, rois_tmp_hwl_a, rotation_matrix_a)
    Poly_anno = Polygon(tuple(corners_a[[0, 2]].t()[[0, 1, 5, 4]]))

    rois_tmp_xyz_b = box_b[0:3]
    rois_tmp_hwl_b = box_b[3:6]
    rois_tmp_ry_b = box_b[-1]
    h_b, w_b, l_b = rois_tmp_hwl_b
    center_b = torch.tensor([rois_tmp_xyz_b[0], rois_tmp_xyz_b[1] - rois_tmp_hwl_b[0] / 2, rois_tmp_xyz_b[2]]).reshape((3, 1)).cuda().float()
    rotation_matrix_b = torch.tensor((Quaternion(axis=[0, 1, 0], radians=float(rois_tmp_ry_b)) *
                                   Quaternion(axis=[1, 0, 0], radians=np.pi / 2)).rotation_matrix).cuda().float()
    corners_b = corners(center_b, rois_tmp_hwl_b, rotation_matrix_b)
    Poly_subm = Polygon(tuple(corners_b[[0, 2]].t()[[0, 1, 5, 4]]))

    box_inter = Poly_anno.intersection(Poly_subm)
    box_union = Poly_anno.union(Poly_subm)

    if dim == 2:
        return box_inter.area / box_union.area

    else:
        ymax = min(center_a[1], center_b[1])
        ymin = max(center_a[1] - h_a,
                   center_b[1] - h_b)

        inter_vol = box_inter.area * max(0, ymax - ymin)
        anno_vol = w_a * l_a * h_a
        subm_vol = w_b * l_b * h_b

        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)

    return float(overlap)


def estimateOverlap(box_a, box_b, dim=2) -> float:
    if box_a == box_b:
        return 1.0

    Poly_anno = fromBoxToPoly(box_a)
    Poly_subm = fromBoxToPoly(box_b)

    box_inter = Poly_anno.intersection(Poly_subm)
    box_union = Poly_anno.union(Poly_subm)
    if dim == 2:
        return box_inter.area / box_union.area

    else:

        ymax = min(box_a.center[1], box_b.center[1])
        ymin = max(box_a.center[1] - box_a.wlh[2],
                   box_b.center[1] - box_b.wlh[2])

        inter_vol = box_inter.area * max(0, ymax - ymin)
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)

    return overlap


class Success(object):
    """Computes and stores the Success"""

    def __init__(self, n=21, max_overlap=1):
        self.max_overlap = max_overlap
        self.Xaxis = np.linspace(0, self.max_overlap, n)
        self.reset()

    def reset(self):
        self.overlaps = []

    def add_overlap(self, val):
        self.overlaps.append(val)

    @property
    def count(self):
        return len(self.overlaps)

    @property
    def value(self):
        succ = [
            np.sum(i >= thres
                   for i in self.overlaps).astype(float) / self.count
            for thres in self.Xaxis
        ]
        return np.array(succ)

    @property
    def average(self):
        if len(self.overlaps) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_overlap


class Precision(object):
    """Computes and stores the Precision"""

    def __init__(self, n=21, max_accuracy=2):
        self.max_accuracy = max_accuracy
        self.Xaxis = np.linspace(0, self.max_accuracy, n)
        self.reset()

    def reset(self):
        self.accuracies = []

    def add_accuracy(self, val):
        self.accuracies.append(val)

    @property
    def count(self):
        return len(self.accuracies)

    @property
    def value(self):
        prec = [
            np.sum(i <= thres
                   for i in self.accuracies).astype(float) / self.count
            for thres in self.Xaxis
        ]
        return np.array(prec)

    @property
    def average(self):
        if len(self.accuracies) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_accuracy


class Accuracy_Completeness(object):
    """Computes and stores the Accuracy and Completeness"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.accuracy = []
        self.completeness = []

    def update(self, preds, gts):
        acc, comp = acc_comp(preds, gts)
        self.accuracy.append(acc)
        self.completeness.append(comp)

    @property
    def count(self):
        return len(self.completeness)

    @property
    def value(self):
        return self.accuracy[-1], self.completeness[-1]

    @property
    def average(self):
        if(self.count == 0):
            return 0, 0
        return np.mean(self.accuracy), np.mean(self.completeness)
