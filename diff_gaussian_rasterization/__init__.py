#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs = None,
        colors_precomp = None,
        scales = None,
        rotations = None,
        cov3D_precomp = None
    ):

        raster_settings = self.raster_settings

        # Check: shs and colors_precomp should not both be empty/None
        shs_given = shs is not None and (shs.numel() > 0 if shs.dim() > 0 else False)
        colors_given = colors_precomp is not None and (colors_precomp.numel() > 0 if colors_precomp.dim() > 0 else False)
        if (not shs_given and not colors_given) or (shs_given and colors_given):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        # Check: scales+rotations or cov3D_precomp
        scale_rot_given = (scales is not None and (scales.numel() > 0 if scales.dim() > 0 else False)) and                          (rotations is not None and (rotations.numel() > 0 if rotations.dim() > 0 else False))
        cov_given = cov3D_precomp is not None and (cov3D_precomp.numel() > 0 if cov3D_precomp.dim() > 0 else False)
        if (not scale_rot_given and not cov_given) or (scale_rot_given and cov_given):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

    def forward_batch_kernel(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None
    ):
        """
        Batch rasterization using kernel version (single forward call for N cameras).
        
        Parameters:
        - means3D: (P, 3)
        - opacities: (P, 1)
        - shs: (P, M, 3) OR None
        - colors_precomp: (P, 3) OR None
        - scales: (P, 3) OR None
        - rotations: (P, 4) OR None
        - cov3D_precomp: (P, 6) OR None

        Returns:
        - color: (N, 3, H, W)
        - radii: (N, P)
        - invdepths: (N, 1, H, W)
        """
        raster_settings = self.raster_settings

        shs_given = shs is not None and shs.numel() > 0
        colors_given = colors_precomp is not None and colors_precomp.numel() > 0
        if (not shs_given and not colors_given) or (shs_given and colors_given):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')

        scale_rot_given = (scales is not None and scales.numel() > 0) and (rotations is not None and rotations.numel() > 0)
        cov_given = cov3D_precomp is not None and cov3D_precomp.numel() > 0
        if (not scale_rot_given and not cov_given) or (scale_rot_given and cov_given):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        device = means3D.device

        if shs is None or shs.numel() == 0:
            shs = torch.empty((means3D.shape[0], 0, 3), dtype=torch.float32, device=device)
        if colors_precomp is None or colors_precomp.numel() == 0:
            colors_precomp = torch.empty((0,), dtype=torch.float32, device=device)
        if scales is None or scales.numel() == 0:
            scales = torch.empty((means3D.shape[0], 3), dtype=torch.float32, device=device)
        if rotations is None or rotations.numel() == 0:
            rotations = torch.empty((means3D.shape[0], 4), dtype=torch.float32, device=device)
        if cov3D_precomp is None or cov3D_precomp.numel() == 0:
            cov3D_precomp = torch.empty((0,), dtype=torch.float32, device=device)

        _, color, radii, _, _, _, invdepths = _C.rasterize_gaussians_batch_kernel(
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix.contiguous(),
            raster_settings.projmatrix.contiguous(),
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos.contiguous(),
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
        )

        return color, radii, invdepths

    def forward_batch_kernel_compact(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        render_color=True,
        render_depth=True,
        return_radii=False,
    ):
        """
        Batch rasterization with optional outputs to reduce peak memory.

        Returns empty tensors for disabled outputs.
        """
        raster_settings = self.raster_settings

        if not render_color and not render_depth:
            raise Exception('Please enable at least one of render_color or render_depth!')

        shs_given = shs is not None and shs.numel() > 0
        colors_given = colors_precomp is not None and colors_precomp.numel() > 0
        if render_color and ((not shs_given and not colors_given) or (shs_given and colors_given)):
            raise Exception('Please provide exactly one of either SHs or precomputed colors when rendering color!')

        scale_rot_given = (scales is not None and scales.numel() > 0) and (rotations is not None and rotations.numel() > 0)
        cov_given = cov3D_precomp is not None and cov3D_precomp.numel() > 0
        if (not scale_rot_given and not cov_given) or (scale_rot_given and cov_given):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        device = means3D.device

        if shs is None or shs.numel() == 0:
            shs = torch.empty((means3D.shape[0], 0, 3), dtype=torch.float32, device=device)
        if colors_precomp is None or colors_precomp.numel() == 0:
            colors_precomp = torch.empty((0,), dtype=torch.float32, device=device)
        if scales is None or scales.numel() == 0:
            scales = torch.empty((means3D.shape[0], 3), dtype=torch.float32, device=device)
        if rotations is None or rotations.numel() == 0:
            rotations = torch.empty((means3D.shape[0], 4), dtype=torch.float32, device=device)
        if cov3D_precomp is None or cov3D_precomp.numel() == 0:
            cov3D_precomp = torch.empty((0,), dtype=torch.float32, device=device)

        _, color, radii, _, _, _, invdepths = _C.rasterize_gaussians_batch_kernel_compact(
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix.contiguous(),
            raster_settings.projmatrix.contiguous(),
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos.contiguous(),
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
            render_color,
            render_depth,
            return_radii,
        )

        return color, radii, invdepths
