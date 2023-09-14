import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import tarfile
import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from ..utils.io import read_image, write_image

logger = logging.getLogger(__name__)

try:
    from redact.settings import Settings
    import redact.v3 as redact
except ImportError as e:
    logger.error('Could not import Brighter.AI API - did you install it?\n'
                 'pip install git+https://github.com/brighter-ai/redact-client.git')
    redact = Settings = None


def blur_detections(image: np.ndarray,
                    faces: List[Tuple[float]],
                    blend_ksize_multiplier: float = 0.2) -> Tuple[np.ndarray]:
    hi, wi = image.shape[:2]
    mask = np.full((hi, wi), 0.0)
    if len(faces) == 0:
        return image, mask

    blurred = image.copy()
    tmp = np.zeros_like(image)
    for bbox in faces:
        x1, y1 = np.floor(bbox[:2]).astype(int)
        x2, y2 = np.ceil(bbox[2:]).astype(int) + 1
        raw_w, raw_h = x2 - x1, y2 - y1
        blend_ksize = int(np.ceil(blend_ksize_multiplier * max(raw_w, raw_h)))
        if blend_ksize % 2 == 1:
            blend_ksize += 1

        # the box corners are sometimes out of the image boundaries
        w = raw_w + 3 * blend_ksize
        h = raw_h + 3 * blend_ksize
        padding = blend_ksize + blend_ksize // 2
        x1, y1 = x1 - padding, y1 - padding
        x2, y2 = x2 + padding, y2 + padding
        x1_, y1_ = max(x1, 0), max(y1, 0)
        x2_, y2_ = min(x2, wi), min(y2, hi)
        patch = np.s_[y1_:y2_, x1_:x2_]

        # define the mask as an ellipse in the bbox
        # extend ellipse to compensate for post-blurring
        dx = ((np.arange(x2 - x1) * 2 - w) / (raw_w + blend_ksize // 2)) ** 2
        dy = ((np.arange(y2 - y1) * 2 - h) / (raw_h + blend_ksize // 2)) ** 2
        mask_box = (dx[None] + dy[:, None]) <= 1
        if blend_ksize > 0:
            mask_box = cv2.blur(mask_box.astype(float), (blend_ksize,) * 2)
        # Crop the mask such that it is still centered.
        mask_box = mask_box[y1_ - y1 : y2_ - y1, x1_ - x1 : x2_ - x1]
        mask[patch] = np.clip(mask[patch] + mask_box, 0, 1)

        # blur a patch bigger than the bbox
        # the kernel size depends on the original bbox size to avoid artifacts
        ksize = max(raw_w, raw_h) // 2 + 1
        blur = cv2.blur(image[patch], (ksize,)*2)

        # copy the masked blur
        # hacky since the mask is over the bbox but the blur is over a larger patch
        tmp[patch] = blur
        if len(image.shape) == 3:
            mask_box = mask_box[..., None]
        blurred[patch] = mask_box * tmp[patch] + (1 - mask_box) * blurred[patch]
        # np.where(mask_box, tmp[patch], blurred[patch])

    blurred = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)

    return blurred, mask


def score_threshold(area: float,
                    score_min: float = 0.3,
                    score_max: float = 0.9,
                    area_min: float = 0.001,
                    area_max: float = 0.15) -> float:
    t = np.log(area / area_min) / np.log(area_max / area_min)
    t = np.clip(t, a_min=0, a_max=1)
    return score_min + (score_max - score_min) * t


def get_box_area_ratio(bbox, image_shape):
    mx, my, Mx, My = bbox
    return (Mx - mx + 1) * (My - my + 1) / (image_shape[0] * image_shape[1])


class BaseAnonymizer:
    labels_filename = 'labels.json'
    min_face_score = None
    max_face_score = None

    def face_is_valid(self, face, image_shape):
        if not isinstance(face, dict):
            face = face.dict()
        if face['score'] is None:
            logger.warning('Found face with score=None.')
            return True
        area_ratio = get_box_area_ratio(face['bounding_box'], image_shape)
        return face['score'] > score_threshold(
            area_ratio, score_min=self.min_face_score,
            score_max=self.max_face_score)


class EgoBlurAnonymizer(BaseAnonymizer):
    min_face_score = 0.4
    max_face_score = 0.9
    min_lp_score = 0.85
    max_lp_score = 0.97

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        hub_dir = torch.hub.get_dir()
        face_path = Path(hub_dir, 'ego_blur_face.jit')
        if not face_path.exists():
            raise FileNotFoundError(
                f'Could not find the EgoBlur Face model at {face_path}. '
                'Download it from https://www.projectaria.com/tools/egoblur/')
        lp_path = Path(hub_dir, 'ego_blur_lp.jit')
        if not face_path.exists():
            raise FileNotFoundError(
                f'Could not find the EgoBlur License Plate model at {lp_path}. '
                'Download it from https://www.projectaria.com/tools/egoblur/')
        self.face_detector = torch.jit.load(face_path, map_location='cpu').to(self.device).eval()
        self.lp_detector = torch.jit.load(lp_path, map_location='cpu').to(self.device).eval()

    def get_detections(self, detector, image_tensor: torch.Tensor,
                       nms_iou_threshold: float = 0.3, max_image_size: Optional[int] = None):
        size = image_tensor.shape[-2:]
        scale = None
        if max_image_size is not None and max(size) > max_image_size:
            scale = max_image_size / max(size)
            size_new = [int(side*scale) for side in size]
            image_tensor = torchvision.transforms.functional.resize(
                image_tensor, size_new, antialias=True)
        with torch.no_grad():
            detections = detector(image_tensor)
        boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims
        nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
        boxes = boxes[nms_keep_idx]
        scores = scores[nms_keep_idx]
        if scale is not None:
            boxes /= scale
        boxes = boxes.cpu().numpy().tolist()
        scores = scores.cpu().numpy().tolist()
        return [dict(bounding_box=b, score=s) for b, s in zip(boxes, scores)]

    def lp_is_valid(self, lp, image_shape):
        area_ratio = get_box_area_ratio(lp['bounding_box'], image_shape)
        return lp['score'] > score_threshold(
            area_ratio, score_min=self.min_lp_score, score_max=self.max_lp_score)

    def blur_image_group(self, input_paths: List[Path], tmp_dir: Path,
                         output_paths: Optional[List[Path]] = None):
        labels_path = tmp_dir / self.labels_filename
        if labels_path.exists():
            logger.info('Reading labels %s', labels_path)
            labels_cached = json.loads(labels_path.read_text())['frames']
            assert len(input_paths) == len(labels_cached), (tmp_dir, len(input_paths))
            labels = None
        else:
            labels_cached = None
            labels = []

        inplace = output_paths is None

        counts = {'faces': 0, 'plates': 0}
        # actually blur the images
        for idx, image_path in enumerate(tqdm(input_paths)):
            image = read_image(image_path)
            if labels_cached is None:
                image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).flip(0)
                image_tensor = image_tensor.to(self.device)
                faces = self.get_detections(self.face_detector, image_tensor)
                plates = self.get_detections(self.lp_detector, image_tensor,
                                             max_image_size=640)
                labels.append(dict(faces=faces, license_plates=plates))
            else:
                faces = labels_cached[idx]['faces']
                plates = labels_cached[idx]['license_plates']

            faces = [f for f in faces if self.face_is_valid(f, image.shape)]
            plates = [f for f in plates if self.lp_is_valid(f, image.shape)]

            blurred, _ = blur_detections(image, [f['bounding_box'] for f in faces])
            blurred, _ = blur_detections(
                blurred, [f['bounding_box'] for f in plates], blend_ksize_multiplier=0)

            assert blurred.shape == image.shape and blurred.dtype == image.dtype
            counts['faces'] += len(faces)
            counts['plates'] += len(plates)

            if inplace:
                if len(faces) > 0 or len(plates) > 0:
                    write_image(image_path, blurred)
            else:
                out = output_paths[idx]
                out.parent.mkdir(exist_ok=True, parents=True)
                write_image(out, blurred)
        if labels is not None:
            labels_path.write_text(json.dumps(dict(frames=labels)))
        logger.info('Finished anonymization in %s', tmp_dir)
        return counts


class BrighterAIAnonymizer(BaseAnonymizer):
    min_face_score = 0.3
    max_face_score = 0.98
    min_lp_score = 0.6

    def __init__(self, apikey, **kwargs):
        self.apikey = apikey
        self.url = Settings().redact_online_url
        self.args = redact.JobArguments(**kwargs)

    def query_frame_labels(self, paths, tmp_dir):
        labels_path = tmp_dir / self.labels_filename
        if labels_path.exists():
            logger.info('Reading labels %s', labels_path)
            with open(labels_path, 'r') as fid:
                labels = redact.JobLabels.parse_raw(fid.read())
            return labels
        logger.info('Calling API with %d images in %s.', len(paths), tmp_dir)

        tmp_dir.mkdir(exist_ok=True, parents=True)
        tar_path = tmp_dir / f'{tmp_dir.name}.tar'
        with tarfile.open(tar_path, 'w') as tar:
            for i, p in enumerate(paths):
                tar.add(p, arcname=f'{i:010}{p.suffix}')

        # call the anonymization API
        instance = redact.RedactInstance.create(
            service='blur', out_type='archives', redact_url=self.url, api_key=self.apikey)
        with open(tar_path, 'rb') as fid:
            job = instance.start_job(file=fid, job_args=self.args)
            job.wait_until_finished()
        status = job.get_status()
        if status.state == redact.JobState.failed:
            logger.warning('Job failed for %s: %s, %s', tmp_dir.name, status.warnings, status.error)
            return None
        labels = job.get_labels()

        with open(labels_path, 'w') as fid:
            fid.write(labels.json())
        tar_path.unlink()
        return labels

    def blur_image_group(self, input_paths: List[Path], tmp_dir: Path,
                         output_paths: Optional[List[Path]] = None):
        labels = self.query_frame_labels(input_paths, tmp_dir)
        if labels is None:
            return None
        inplace = output_paths is None

        assert len(input_paths) == len(labels.frames), (tmp_dir, len(input_paths))
        counts = {'faces': 0, 'plates': 0}
        # actually blur the images
        for idx, image_path in enumerate(input_paths):
            label = labels.frames[idx]
            assert label.index == (idx+1)
            if not label.faces and not label.license_plates and inplace:
                continue

            image = read_image(image_path)
            faces = [f for f in label.faces if self.face_is_valid(f, image.shape)]
            plates = [f for f in label.license_plates if f.score >= self.min_lp_score]

            blurred, _ = blur_detections(image, [f.bounding_box for f in faces])
            blurred, _ = blur_detections(
                blurred, [f.bounding_box for f in plates], blend_ksize_multiplier=0)

            assert blurred.shape == image.shape and blurred.dtype == image.dtype
            counts['faces'] += len(faces)
            counts['plates'] += len(plates)

            if inplace:
                if len(faces) > 0 or len(plates) > 0:
                    write_image(image_path, blurred)
            else:
                out = output_paths[idx]
                out.parent.mkdir(exist_ok=True, parents=True)
                write_image(out, blurred)
        logger.info('Finished anonymization in %s', tmp_dir)
        return counts
