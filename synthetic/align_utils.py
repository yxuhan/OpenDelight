import numpy as np
import cv2


def _compute_ffhq_quad(lm: np.ndarray, border_scale: float = 1.10):
    lm = np.asarray(lm, dtype=np.float32).reshape(-1, 2)

    lm_eye_left   = lm[36:42]
    lm_eye_right  = lm[42:48]
    lm_mouth_out  = lm[48:60]

    eye_left  = lm_eye_left.mean(axis=0)
    eye_right = lm_eye_right.mean(axis=0)
    eye_avg   = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left

    mouth_left  = lm_mouth_out[0]
    mouth_right = lm_mouth_out[6]
    mouth_avg   = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * np.array([-1, 1], dtype=np.float32)
    x = x / (np.hypot(x[0], x[1]) + 1e-8)
    x = x * max(np.hypot(*(eye_to_eye)) * 2.0, np.hypot(*(eye_to_mouth)) * 1.8)

    y = np.flipud(x) * np.array([-1, 1], dtype=np.float32)
    c = eye_avg + eye_to_mouth * 0.1

    x = x * border_scale
    y = y * border_scale

    quad = np.stack([c - x - y,
                     c - x + y,
                     c + x + y,
                     c + x - y], axis=0).astype(np.float32)
    return quad, c, x, y


def align_face(
    img: np.ndarray,
    output_size: int,
    lm: np.ndarray,
    border_scale: float = 1.0,
    interpolation: int = cv2.INTER_CUBIC,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value =  0,
):
    assert img.ndim == 3 and img.shape[2] in (1, 3, 4), "img must be (H,W,C)"
    quad, _, _, _ = _compute_ffhq_quad(lm, border_scale=border_scale)

    dst = np.array([[0, 0],
                    [0, output_size],
                    [output_size, output_size],
                    [output_size, 0]], dtype=np.float32)

    H_src2dst = cv2.getPerspectiveTransform(quad, dst).astype(np.float64)

    aligned = cv2.warpPerspective(
        img, H_src2dst, dsize=(output_size, output_size),
        flags=interpolation, borderMode=border_mode, borderValue=border_value
    )

    return aligned.clip(0, 255), H_src2dst


def inverse_align_face(
    aligned_img: np.ndarray,
    H_src2dst: np.ndarray,
    orig_size: tuple[int, int],
    interpolation: int = cv2.INTER_CUBIC,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value =  0,
):
    H_dst2src = np.linalg.inv(H_src2dst)

    W, H = orig_size  # OpenCV 的 dsize=(W,H)
    img_reprojected = cv2.warpPerspective(
        aligned_img, H_dst2src, dsize=(W, H),
        flags=interpolation, borderMode=border_mode, borderValue=border_value
    )
    return img_reprojected.clip(0, 255)
