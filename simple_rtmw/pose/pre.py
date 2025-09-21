
import cv2
import numpy as np


def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale).

    Args:
        bbox: Bounding box(es) in shape (4,) or (n, 4), formatted as (left, top, right, bottom)
        padding: BBox padding factor that will be multilied to scale. Default: 1.0

    Returns:
        Center (x, y) of the bbox in shape (2,) or (n, 2)
        Scale (w, h) of the bbox in shape (2,) or (n, 2)
    """
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]  # convert single bbox from (4, ) to (1, 4)

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _get_point_3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center: Center of the bounding box (x, y).
        scale: Scale of the bounding box wrt [width, height].
        output_size: Size of the destination heatmaps.

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # transform direction vector
    src_dir = np.array([0., src_w * -0.5])
    dst_dir = np.array([0., dst_w * -0.5])

    # src points (refer to MMPose/HRNet affine formulation)
    # The first point is the bbox center; the second is shifted by the
    # direction vector corresponding to half bbox width; the third is
    # derived to form a right-angled triangle.
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = _get_point_3(src[0, :], src[1, :])

    # dst points
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]

    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_point_3(dst[0, :], dst[1, :])

    # affine transform matrix
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))  # type: ignore


def top_down_affine(
    input_size: np.ndarray,
    bbox_scale: np.ndarray,
    bbox_center: np.ndarray,
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by anp.ndarray.

    Args:
        input_size: The input size of the model.
        bbox_scale: The bbox scale of the img.
        bbox_center: The bbox center of the img.
        img: The original image.

    Returns:
        img: Image after affine transform.
        bbox_scale: Bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    aspect_ratio = w / h
    b_w, b_h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        b_w > b_h * aspect_ratio,
        np.hstack([b_w, b_w / aspect_ratio]),
        np.hstack([b_h * aspect_ratio, b_h]),
    )

    # get the affine matrix
    warp_mat = get_warp_matrix(
        bbox_center,
        bbox_scale,
        output_size=(w, h),
    )

    # do affine transform
    img = cv2.warpAffine(
        img,
        warp_mat,
        warp_size,
        flags=cv2.INTER_LINEAR,
    )

    return img, bbox_scale
