import unittest

import numpy as np

from zerors.attention import prepare_heatmap, region_grow, salient_regions
from zerors.config import PromptRefinementConfig
from zerors.point_box_refiner import PointBoxPromptRefiner


class FakeSam:
    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        mask = np.zeros(self.shape, dtype=bool)
        if box is not None:
            x1, y1, x2, y2 = box[0].astype(int)
            mask[y1 : y2 + 1, x1 : x2 + 1] = True
        else:
            x, y = point_coords[0].astype(int)
            mask[y, x] = True
        return np.stack([mask, mask, mask]), np.array([0.2, 0.9, 0.1]), None


class PointBoxRefinerTest(unittest.TestCase):
    def test_saliency_to_point_box_prompt(self):
        attention = np.zeros((8, 8), dtype=np.float32)
        attention[3:6, 2:5] = 1.0
        config = PromptRefinementConfig(gaussian_sigma=0.1, min_component_area=1, point_stride=1)
        heatmap = prepare_heatmap(attention, (32, 32), config.gaussian_sigma)
        regions = salient_regions(heatmap, config)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].points.shape[1], 2)
        self.assertGreater(regions[0].box[2], regions[0].box[0])

    def test_refiner_returns_best_sam_mask(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        attention = np.zeros((8, 8), dtype=np.float32)
        attention[3:6, 2:5] = 1.0
        config = PromptRefinementConfig(gaussian_sigma=0.1, min_component_area=1)
        result = PointBoxPromptRefiner(FakeSam(), config).refine(image, attention)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.9)
        self.assertTrue(result[0].mask.any())

    def test_region_grow_stays_inside_threshold(self):
        heatmap = np.zeros((5, 5), dtype=np.float32)
        heatmap[1:4, 1:4] = 1.0
        mask = region_grow(heatmap, [(2, 2)], threshold=0.3)
        self.assertEqual(mask.sum(), 9)


if __name__ == "__main__":
    unittest.main()
