
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
import copy
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

############### Image-Level discriminator ##############
class Discriminator_img(nn.Module):
    def __init__(self, input_size, ndf1=256, ndf2=128):
        super(Discriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(input_size, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = torch.flatten(x)
        return x

############### Instance-Level Attention discriminator ##############
class Discriminator_inst(nn.Module):
    def __init__(self, output_size = 1, num_head = 1, flat = 1):
        super(Discriminator_inst, self).__init__()
        self.flat = flat
        self.drop = nn.Dropout(p = 0.3)
        self.atten = nn.MultiheadAttention(512, num_head)

        self.linf = nn.Sequential(nn.LayerNorm((512,)),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.Dropout(.3),
            nn.LayerNorm((128,)),
            nn.Linear(128, output_size))
        

    def forward(self, q, k, v):
        q = self.drop(q)
        v = self.drop(v)
        q = self.atten(q, k, v, need_weights=False)[0]
        q = self.linf(q)
        if self.flat:
            q = torch.flatten(q)
        return q

############### Embedding Layer ##############
class Embedding(nn.Module):
    def __init__(self, num_classes):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, 512)

    def forward(self, x):
        x = self.embedding(x)
        return x

############### Projection Layer ##############
class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.project = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.project(x)
        return x

############### Gradient reverse function ###############
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

############### Our ACIA Model ###############
@META_ARCH_REGISTRY.register()
class ACIAMSDAGeneralizedRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        num_classes: int
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.num_classes = num_classes

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.dis_type = dis_type
  
        self.D_img = Discriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device)  # Need to know the channel
        self.class_disc = None
        self.embedding = None
        self.projection = None
        self.projection2 = None
        self.build_class_disc()

    def build_discriminator(self):
        self.D_img = Discriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    def build_class_disc(self):
        self.class_disc = Discriminator_inst(output_size=3, num_head=4, flat = 0).to(self.device)
        self.embedding = Embedding(self.num_classes).to(self.device)
        self.projection = Projection().to(self.device)
        self.projection2 = Projection().to(self.device)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def convert_gt_to_rcn(self, gt):
        temp = copy.deepcopy(gt)
        for item in temp:
            item.set('objectness_logits', torch.ones(len(item)).to(self.device))
            item.set('proposal_boxes', item.get('gt_boxes'))
            item.remove('gt_classes')
            item.remove('gt_boxes')
        return temp

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, target_type = 1
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # Only used for Testing
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        source_label = target_type
        target_label = 3

        # Image-Level Alignment
        if branch == "image_level_align":
            images_s, images_t = self.preprocess_image_train(batched_inputs)
            features = self.backbone(images_s.tensor)
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            criterion = torch.nn.CrossEntropyLoss()
            loss_D_img_s = criterion(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))/D_img_out_s.shape[0]
            loss_D_img_t = 0
            # Only use target data one time
            if (target_type == 2):
                features_t = self.backbone(images_t.tensor)
                features_t = grad_reverse(features_t[self.dis_type])
                D_img_out_t = self.D_img(features_t)
                loss_D_img_t = criterion(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))/D_img_out_t.shape[0]

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s * .01
            losses["loss_D_img_t"] = loss_D_img_t * .01
            return losses, [], [], None

        # Instance-Level Alignment
        elif(branch == "instance_level_align"):
            if (self.class_disc == None or self.embedding == None):
                self.build_class_disc()

            source1, source2 = batched_inputs
            images_s1 = self.preprocess_image(source1)
            images_s2 = self.preprocess_image(source2)

            gt_instances_s1 = [x["instances"].to(self.device) for x in source1]
            gt_instances_s2 = [x["instances"].to(self.device) for x in source2]

            features_s1 = self.backbone(images_s1.tensor)
            features_s2 = self.backbone(images_s2.tensor)

            # Change GT to RPN type, expected by detectron2
            proposals_rpn_s1 = self.convert_gt_to_rcn(gt_instances_s1)
            proposals_rpn_s2 = self.convert_gt_to_rcn(gt_instances_s2)

            gt_labels_s1 = gt_instances_s1[0].gt_classes
            gt_labels_s2 = gt_instances_s2[0].gt_classes
            if(len(gt_instances_s1) == 2):
                gt_labels_s1 = torch.cat((gt_labels_s1, gt_instances_s1[1].gt_classes))
                gt_labels_s2 = torch.cat((gt_labels_s2, gt_instances_s2[1].gt_classes))

            # Getting class-embedding output
            embed_s1 = self.embedding(gt_labels_s1)
            embed_s2 = self.embedding(gt_labels_s2)

            # Output is box feature only, due to argument branch.. Check roi_heads code...
            box_features_s1 = self.roi_heads(
                images_s1,
                features_s1,
                proposals_rpn_s1,
                compute_loss=True,
                targets=gt_instances_s1,
                branch=branch,
            )

            box_features_s2 = self.roi_heads(
                images_s2,
                features_s2,
                proposals_rpn_s2,
                compute_loss=True,
                targets=gt_instances_s2,
                branch=branch,
            )

            # Creating 2 projections for attention
            q1 = self.projection(grad_reverse(box_features_s1))
            v1 = self.projection2(grad_reverse(box_features_s1))

            q2 = self.projection(grad_reverse(box_features_s2))
            v2 = self.projection2(grad_reverse(box_features_s2))

            # Using attention
            box_features_s1 = self.class_disc(q1, embed_s1, v1)
            box_features_s2 = self.class_disc(q2, embed_s2, v2)

            output_s1 = torch.FloatTensor(box_features_s1.shape).fill_(0).to(self.device)
            output_s2 = torch.FloatTensor(box_features_s2.shape).fill_(1).to(self.device)

            # Change to Cross-Entropy if number of source is more than 2.
            criterion = torch.nn.BCEWithLogitsLoss()

            loss = 0
            loss += criterion(box_features_s1, output_s1)
            loss += criterion(box_features_s2, output_s2)

            return loss/2.
        
        
        # If not instance or image level alignment
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)

        # Source Data Training
        if branch == "supervised":
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            source_label = target_type
            criterion = torch.nn.CrossEntropyLoss()
            # This step is done to avoid the DDP problem... Check engine -> trainer.py for more detail..
            loss_D_img_s = criterion(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))/1000.0
             
            # RPN
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # ROI Heads
            _, detector_losses, box_features = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            
            losses["loss_D_img_s"] = loss_D_img_s * 0.0001
            return losses, [], [], None

        # Target Data Training
        elif branch == "supervised_target":
            # RPN
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # ROI Heads
            _, detector_losses, _ = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        # This is used to generate Psuedo-labels by teacher model on target domain
        elif branch == "unsup_data_weak":
            # RPN
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # ROI Heads
            # No GT labels are used!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            return {}, proposals_rpn, proposals_roih, ROI_predictions
        # If branch not found!
        else:
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


