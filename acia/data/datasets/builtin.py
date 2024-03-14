# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
# from fvcore.common.file_io import PathManager
from iopath.common.file_io import PathManager

from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .cityscapes_foggy import load_cityscapes_instances
import io
from acia.data.datasets.cityscape_car import load_cityscapes_instances_car
from .kitty_loader import register_kitty
from acia.data.datasets.coco import my_register_coco_instances
import logging
from .voc_loader import register_bdd
from .voc_loader_mixed import register_bdd_mixed
from .voc_loader_car import register_bdd_car
from acia.data.datasets.synscape import synapse_register
from .cityscapes_mixed import load_cityscapes_instances_mixed


logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


_root = 'datasets/'
register_coco_unlabel(_root)

def register_coco_new():
    my_register_coco_instances("filter_coco",
                        {},
                        "datasets/filter2.json",
                        "datasets/train2017")
    MetadataCatalog.get("filter_coco").evaluator_type = "coco"

# ==== Predefined splits for raw cityscapes foggy images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_foggy_train": ("cityscapes_foggy/leftImg8bit/train/", "cityscapes_foggy/gtFine/train/"),
    "cityscapes_foggy_val": ("cityscapes_foggy/leftImg8bit/val/", "cityscapes_foggy/gtFine/val/"),
    "cityscapes_foggy_test": ("cityscapes_foggy/leftImg8bit/test/", "cityscapes_foggy/gtFine/test/"),
}


def register_all_cityscapes_foggy(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=False, to_polygons=False
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="coco", **meta
        )

def register_cityscapes_mixed(root="datasets/"):
    key = "cityscapes_mixed"
    image_dir = "cityscapes/leftImg8bit/train/"
    gt_dir = "cityscapes/gtFine/train/"
    meta = _get_builtin_metadata("cityscapes")
    image_dir = os.path.join(root, image_dir)
    gt_dir = os.path.join(root, gt_dir)

    inst_key = key

    DatasetCatalog.register(
        inst_key,
        lambda x=image_dir, y=gt_dir: load_cityscapes_instances_mixed(
            x, y, from_json=False, to_polygons=False
        ),
    )
    MetadataCatalog.get(inst_key).set(
        image_dir=image_dir, gt_dir=gt_dir, evaluator_type="coco", thing_classes = ["person", "rider","car", "truck", "bus", "train", "motor", "bike"]
    )


# ==== Predefined splits for Clipart (PASCAL VOC format) ===========
def register_all_clipart(root):
    SPLITS = [
        ("Clipart1k_train", "clipart", "train"),
        ("Clipart1k_test", "clipart", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
    
def register_all_VOC2012(root):
    SPLITS = [
        ("voc_2012_trainval", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

def register_all_VOC2007(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_daytime(root):
    SPLITS = [
        ("Daytime", "Daytime", "trainval"),
        ("Daytime_train", "Daytime", "train"),
        ("Daytime_val", "Daytime", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_bdd(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_bdd"

def register_all_daytime_car(root):
    SPLITS = [
        ("Daytime_car", "Daytime", "trainval"),
        ("Daytime_car_train", "Daytime", "train"),
        ("Daytime_car_val", "Daytime", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_bdd_car(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_bdd_car"



def register_cityscapes_car(root = "datasets"):
    key = "cityscapes_car"
    image_dir = "cityscapes/leftImg8bit/train/"
    gt_dir = "cityscapes/gtFine/train/"
    meta = {'thing_classes': ['car',]}
    image_dir = os.path.join(root, image_dir)
    gt_dir = os.path.join(root, gt_dir)

    inst_key = key
    DatasetCatalog.register(
        inst_key,
        lambda x=image_dir, y=gt_dir: load_cityscapes_instances_car(
            x, y, from_json=True, to_polygons=True
        ),
    )
    MetadataCatalog.get(inst_key).set(
        image_dir=image_dir, gt_dir=gt_dir, evaluator_type="coco", **meta
    )

def register_all_night(root):
    SPLITS = [
        ("Night", "Night", "trainval"),
        ("Night_train", "Night", "train"),
        ("Night_val", "Night", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_bdd(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_bdd"

def register_all_duskdawn(root):
    SPLITS = [
        ("DuskDawn", "DuskDawn", "trainval"),
        ("DuskDawn_train", "DuskDawn", "train"),
        ("DuskDawn_val", "DuskDawn", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_bdd(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_bdd"

def register_all_daytime_mixed(root):
    SPLITS = [
        ("Daytime_mixed", "Daytime", "trainval"),
        ("Daytime_mixed_train", "Daytime", "train"),
        ("Daytime_mixed_val", "Daytime", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_bdd_mixed(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_bdd_mixed"

def register_all_kitty(root = "datasets"):
    SPLITS = [
        ("Kitty", "new_kitty", "trainval"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_kitty(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_bdd"

register_coco_new()
register_cityscapes_mixed()
register_cityscapes_car()
synapse_register("synscapes", {}, "datasets/Synscapes/meta", "datasets/Synscapes/img/rgb")
register_all_cityscapes_foggy(_root)
root2 = 'datasets/'
register_all_daytime(root2)
register_all_daytime_car(root2)
register_all_night(root2)
register_all_duskdawn(root2)
register_all_daytime_mixed(root2)
register_all_kitty()

