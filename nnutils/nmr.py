from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tqdm

import torch

import neural_renderer

from nnutils import geom_utils
from scipy.spatial.transform import Rotation as R

#############
### Utils ###
#############
def convert_as(src, trg):
    return src.to(trg.device).type_as(trg)

class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, img_size=256):
        super(NeuralRenderer, self).__init__()
        self.renderer = neural_renderer.Renderer()

        # Adjust the core renderer
        self.renderer.image_size = img_size
        self.renderer.perspective = False

        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.eye = [0, 0, -2.732]

        # Make it a bit brighter for vis
        self.renderer.light_intensity_ambient = 0.8

        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.light_intensity_ambient = 1
        self.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        self.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def create_intrinsic_matrix(self, scale):
        K = torch.tensor([[scale, 0, 0],
                          [0, scale, 0],
                          [0, 0, 1]], dtype=torch.float32, device=scale.device)  # Ensure same device
        return K[None, :, :]  # Add a batch dimension

    def forward(self, vertices, faces, cams, textures=None):
        # R = cams[:, 3:7]  # Extract rotation
        # t = cams[:, 1:3]  # Extract translation
        # scale = cams[:, 0]  # Extract scale or other factor (if applicable)

        if cams.shape[1] == 7:
            quat = cams[:, -4:]  # Quaternion
            scale = cams[:, 0].contiguous().view(-1, 1, 1)  # Scale
            trans = cams[:, 1:3].contiguous().view(cams.size(0), 1, -1)  # Translation

            # Convert quaternion to rotation matrix
            R_matrix = R.from_quat(quat.detach().cpu().numpy()).as_matrix()
            R_matrix = torch.tensor(R_matrix, dtype=torch.float32, device=cams.device)  # Convert back to torch tensor
            t = trans  # Use the translation 
            t = torch.cat([t, torch.zeros(t.shape[0], 1, 1, device=cams.device)], dim=2)  # Add a zero for z

            t = t.float()  # Ensure translation tensor is Float
            vertices = vertices.float()
            K = self.create_intrinsic_matrix(scale)

            print("Rotation Matrix (R):", R_matrix)  # Debugging output
            print("Translation (t):", t)  # Debugging output
        else:
            raise ValueError("Invalid cam shape: expected shape (B, 7)")


        verts = self.proj_fn(vertices, cams)
        vs = verts.clone()
        vs[:, :, 1] *= -1
        fs = faces.clone()
        if textures is None:
            self.mask_only = True
            masks = self.renderer.render_silhouettes(vs,fs, R=R_matrix, t=t, K=K)
            return masks
        else:
            self.mask_only = False
            ts = textures.clone()
            imgs = self.renderer.render(vs, fs, ts,  R=R_matrix, t=t, K=K)[0] #only keep rgb, no alpha and depth
            return imgs
        


############# TESTS #############
def exec_main():
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    vertices, faces = neural_renderer.load_obj(obj_file)

    renderer = NMR()
    renderer.to_gpu(device=0)

    masks = renderer.forward_mask(vertices[None, :, :], faces[None, :, :])
    print(np.sum(masks))
    print(masks.shape)

    grad_masks = masks*0 + 1
    vert_grad = renderer.backward_mask(grad_masks)
    print(np.sum(vert_grad))
    print(vert_grad.shape)

    # Torch API
    mask_renderer = NeuralRenderer()
    vertices_var = torch.autograd.Variable(torch.from_numpy(vertices[None, :, :]).cuda(device=0), requires_grad=True)
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
    
    for ix in range(100):
        masks_torch = mask_renderer.forward(vertices_var, faces_var)
        vertices_var.grad = None
        masks_torch.backward(torch.from_numpy(grad_masks).cuda(device=0))

    print(torch.sum(masks_torch))
    print(masks_torch.shape)
    print(torch.sum(vertices_var.grad))

# @DeprecationWarning
def teapot_deform_test():
    #
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    img_file = 'birds3d/external/neural_renderer/examples/data/example2_ref.png'
    img_save_dir = 'birds3d/cachedir/nmr/'

    vertices, faces = neural_renderer.load_obj(obj_file)
    from skimage.io import imread
    image_ref = imread(img_file).astype('float32').mean(-1) / 255.
    image_ref = torch.autograd.Variable(torch.Tensor(image_ref[None, :, :]).cuda(device=0))

    mask_renderer = NeuralRenderer()
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
    cams = np.array([1., 0, 0, 1, 0, 0, 0], dtype=np.float32)
    cams_var = torch.autograd.Variable(torch.from_numpy(cams[None, :]).cuda(device=0))
    print(cams_var)

    class TeapotModel(torch.nn.Module):
        def __init__(self):
            super(TeapotModel, self).__init__()
            vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
            self.vertices_var = torch.nn.Parameter(vertices_var)

        def forward(self):
            return mask_renderer.forward(self.vertices_var, faces_var, cams_var)

    opt_model = TeapotModel()


    optimizer = torch.optim.Adam(opt_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    # from time import time
    loop = tqdm.tqdm(range(300))
    print('Optimizing Vertices: ')
    for ix in loop:
        # t0 = time()
        optimizer.zero_grad()
        masks_pred = opt_model()
        loss = torch.nn.MSELoss()(masks_pred, image_ref)
        loss.backward()
        if ix % 20 == 0:
            im_rendered = masks_pred.data[0, :, :]
            scipy.misc.imsave(img_save_dir + 'iter_{}.png'.format(ix), im_rendered)
        optimizer.step()
        # t1 = time()
        # print('one step %g sec' % (t1-t0))

if __name__ == '__main__':
    # exec_main()
    teapot_deform_test()
