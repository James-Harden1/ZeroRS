import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import random_color
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.engine.defaults import get_model_from_module


class OdiseRefiner:
    def __init__(self, config_file: str, weights_file: str) -> None:
        self.cfg = LazyConfig.load(config_file)
        self.cfg.model.overlap_threshold = 0
        self.cfg.model.clip_head.alpha = 0.15
        self.cfg.model.clip_head.beta = 0.85
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = instantiate_odise(self.cfg.model).to(self.device)
        ODISECheckpointer(self.base_model).load(weights_file)
        self.wrapper_cfg = self.cfg.dataloader.wrapper
        wrapper = self.wrapper_cfg
        while "model" in wrapper:
            wrapper = wrapper.model
        wrapper.model = get_model_from_module(self.base_model)
        self.predictor = None

    def update_vocabulary(self, labels: list[str]) -> None:
        labels = labels or ["object"]
        metadata_name = "zerors_odise_runtime"
        if metadata_name in MetadataCatalog:
            MetadataCatalog.remove(metadata_name)
        metadata = MetadataCatalog.get(metadata_name)
        metadata.thing_classes = labels
        metadata.stuff_classes = []
        metadata.thing_colors = [random_color(rgb=True, maximum=1) for _ in labels]
        metadata.thing_dataset_id_to_contiguous_id = {index: index for index in range(len(labels))}
        metadata.stuff_dataset_id_to_contiguous_id = {}
        self.wrapper_cfg.labels = [[label] for label in labels]
        self.wrapper_cfg.metadata = metadata
        self.predictor = instantiate(self.cfg.dataloader.wrapper).eval()

    def predict_crop(self, image_bgr, labels: list[str]):
        self.update_vocabulary(labels)
        height, width = image_bgr.shape[:2]
        image_rgb = image_bgr[:, :, ::-1]
        tensor = torch.as_tensor(image_rgb.astype("float32").transpose(2, 0, 1))
        with torch.inference_mode():
            return self.predictor([{"image": tensor, "height": height, "width": width}])[0]
