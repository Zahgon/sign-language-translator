"""
This module contains the `MediaPipeLandmarksModel` class, which is a deep learning-based
video embedding model utilizing the MediaPipe framework for extracting pose and hand
landmarks from video frames.

Classes:
    MediaPipeLandmarksModel: A video embedding model that utilizes MediaPipe for pose and hand landmark extraction.

Example:

.. code-block:: python

    from sign_language_translator.models import MediaPipeLandmarksModel
    from sign_language_translator.vision.utils import iter_frames_with_opencv

    mediapipe_model = MediaPipeLandmarksModel(number_of_persons=1)

    frame_sequence = iter_frames_with_opencv("video.mp4")
    embedding = mediapipe_model.embed(frame_sequence, landmark_type="world")
    print(embedding.shape)
"""

from os.path import join
from typing import Dict, Iterable, List, Optional, Union

try:
    import mediapipe
except ImportError:
    mediapipe = None

import numpy as np
import torch
from numpy.typing import NDArray

from sign_language_translator.config.assets import Assets
from sign_language_translator.models.video_embedding.video_embedding_model import (
    VideoEmbeddingModel,
)
from sign_language_translator.utils import ProgressStatusCallback


class MediaPipeLandmarksModel(VideoEmbeddingModel):
    """
    A video embedding model using MediaPipe to extract pose and hand landmarks from video frames.

    Args:
        pose_model_name (str): The name of the pose estimation model.
        hand_model_name (str): The name of the hand estimation model.
        number_of_persons (int): The maximum number of persons to detect in each frame.

    Attributes:
        n_persons (int): The maximum number of persons to detect in each frame.

    Methods:
        embed: Embeds a sequence of frames using pose and hand landmarks.
    """

    def __init__(
        self,
        pose_model_name="pose_landmarker_heavy.task",
        hand_model_name="hand_landmarker.task",
        number_of_persons: int = 1,
    ) -> None:
        if mediapipe is None:
            raise ImportError(
                "The 'mediapipe' package is required to use the 'MediaPipeLandmarksModel'. "
                "Install it using `pip install sign-language-translator[mediapipe]`. "
                "(also make sure if your python version is compatible with mediapipe)."
            )

        self._pose_class = mediapipe.tasks.vision.PoseLandmarker
        self._hand_class = mediapipe.tasks.vision.HandLandmarker

        path = self.__download_and_get_model_path(f"models/mediapipe/{pose_model_name}")
        self._pose_options = mediapipe.tasks.vision.PoseLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=path),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            num_poses=number_of_persons,
        )

        path = self.__download_and_get_model_path(f"models/mediapipe/{hand_model_name}")
        self._hand_options = mediapipe.tasks.vision.HandLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=path),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            num_hands=number_of_persons * 2,
        )

        self.n_persons = number_of_persons

    def embed(
        self,
        frame_sequence: Iterable[Union[torch.Tensor, NDArray[np.uint8]]],
        landmark_type: str = "world" or "image" or "all",
        progress_callback: Optional[ProgressStatusCallback] = None,
        total_frames: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Embed a sequence of frames (video) into a sequence of pose & hand landmarks.

        Args:
            frame_sequence (Iterable[torch.Tensor | NDArray[np.uint8]]): A sequence of video frames as 3D arrays (W, H, c).
            landmark_type (str): The type of landmarks to include in the embedding ("world", "image", "all").

        Returns:
            torch.Tensor: A tensor containing the frame embeddings.
        """
        pass

    def _flatten_landmarks(self, landmarks) -> List[float]:
        pass

    def _extract_from_pose_results(self, pose_result) -> Dict[str, List[List[float]]]:
        pass

    def _extract_from_hand_results(self, hand_result) -> Dict[str, List[List[float]]]:
        pass

    def _arange_pose_and_hands(
        self,
        poses: Dict[str, List[List[float]]],
        hands: Dict[str, List[List[float]]],
    ) -> Dict[str, List[List[float]]]:
        # TODO: Match left & right hands to poses
        # by using minimum distance between hand image centers
        # np.linalg.norm(pose[left_hand_ids].mean(axis=...), hands.mean(axis=...).T).argmin(axis=...)
        pass

    def _create_frame_embedding(
        self, persons: Dict[str, List[List[float]]], landmark_type: str
    ) -> List[float]:
        pass

    def __download_and_get_model_path(self, model_local_path: str):
        Assets.download(
            model_local_path,
            progress_bar=True,
            leave=False,
            chunk_size=1048576,
        )
        return join(Assets.ROOT_DIR, model_local_path)
