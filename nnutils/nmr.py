from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tqdm

import torch

import neural_renderer

from nnutils import geom_utils

#############
### Utils ###
#############
def convert_as(src, trg):
    return src.to(trg.device).type_as(trg)

########################################################################
############ Wrapper class for the chainer Neural Renderer #############
##### All functions must only use numpy arrays as inputs/outputs #######
########################################################################
class NMR(object):
    def __init__(self):
        # setup renderer
        renderer = neural_renderer.Renderer()
        self.renderer = renderer

    def to_gpu(self, device=0):
        # self.renderer.to_gpu(device)
        self.cuda_device = device

    def forward_mask(self, vertices, faces):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
        Returns:
            masks: B X 256 X 256 numpy array
        '''
        self.faces = faces.clone().detach().float().cuda(self.cuda_device).requires_grad_() # chainer.Variable(chainer.cuda.to_gpu(faces, self.cuda_device))
        self.vertices = vertices.clone().detach().float().cuda(self.cuda_device).requires_grad_()
        # import ipdb;ipdb.set_trace()
        self.masks = self.renderer.render_silhouettes(self.vertices, self.faces)

        masks = self.masks.detach()
        return masks
    
    def backward_mask(self, grad_masks):
        ''' Compute gradient of vertices given mask gradients.
        Args:
            grad_masks: B X 256 X 256 numpy array
        Returns:
            grad_vertices: B X N X 3 numpy array
        '''
        # self.masks.grad =  #  chainer.cuda.to_gpu(grad_masks, self.cuda_device)
        print('basckward, mask start')

        self.masks.backward(grad_masks.cuda(self.cuda_device))
        print('basckward, mask end')

        return self.vertices.grad

    def forward_img(self, vertices, faces, textures):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
            textures: B X F X T X T X T X 3 numpy array
        Returns:
            images: B X 3 x 256 X 256 numpy array
        '''
        self.faces =    faces.detach().clone().cuda(self.cuda_device).float().requires_grad_()
        self.vertices = vertices.detach().clone().cuda(self.cuda_device).float().requires_grad_()
        self.textures = textures.detach().clone().cuda(self.cuda_device).float().requires_grad_()
        images = self.renderer.render(self.vertices, self.faces, self.textures)

        images = self.images = images[0] # only keep rgb, no alpha and depth
        return images.detach()


    def backward_img(self, grad_images):
        ''' Compute gradient of vertices given image gradients.
        Args:
            grad_images: B X 3? X 256 X 256 numpy array
        Returns:
            grad_vertices: B X N X 3 numpy array
            grad_textures: B X F X T X T X T X 3 numpy array
        '''
        print('basckward, img start')
        # import ipdb;ipdb.set_trace()
        images_grad =  grad_images.cuda(self.cuda_device)
        print('go to here')
        self.images.backward(images_grad)
        print('basckward, img end')

        return self.vertices.grad, self.textures.grad 

class Render(torch.nn.Module):
    # TODO(Shubham): Make sure the outputs/gradients are on the GPU
    def __init__(self, renderer):
        super(Render, self).__init__()
        self.renderer = renderer

    def forward(self, vertices, faces, textures=None,renderer=None):
        # B x N x 3
        # Flipping the y-axis here to make it align with the image coordinate system!
        self.renderer = renderer
        vs = vertices.clone()
        vs[:, :, 1] *= -1
        fs = faces.clone()
        # import ipdb;ipdb.set_trace()
        if textures is None:
            self.mask_only = True
            masks = self.renderer.forward_mask(vs, fs)
            return convert_as(masks, vertices)
        else:
            self.mask_only = False
            ts = textures#.cpu().numpy()
            imgs = self.renderer.forward_img(vs, fs, ts)
            return convert_as(imgs, vertices)
 


########################################################################
################# Wrapper class a rendering PythonOp ###################
##### All functions must only use torch Tensors as inputs/outputs ######
########################################################################
class Render2(torch.autograd.Function):
    # TODO(Shubham): Make sure the outputs/gradients are on the GPU
    def __init__(self, renderer):
        super(Render, self).__init__()
        self.renderer = renderer

    @staticmethod
    def forward(self, vertices, faces, textures=None,renderer=None):
        # B x N x 3
        # Flipping the y-axis here to make it align with the image coordinate system!
        self.renderer = renderer
        vs = vertices.clone()
        vs[:, :, 1] *= -1
        fs = faces.clone()
        if textures is None:
            self.mask_only = True
            masks = self.renderer.forward_mask(vs, fs)
            return convert_as(masks, vertices)
        else:
            self.mask_only = False
            ts = textures#.cpu().numpy()
            imgs = self.renderer.forward_img(vs, fs, ts)
            return convert_as(imgs, vertices)

    @staticmethod
    def backward(self, grad_out):
        print('basckward,start')
        g_o = grad_out.detach()#.cpu().numpy()
        if self.mask_only:
            grad_verts = self.renderer.backward_mask(g_o)
            grad_verts = convert_as(grad_verts, grad_out)
            grad_tex = None
        else:
            grad_verts, grad_tex = self.renderer.backward_img(g_o)
            grad_verts = convert_as(grad_verts, grad_out)
            grad_tex = convert_as(grad_tex, grad_out)

        grad_verts[:, :, 1] *= -1
        print('basckward end')
        return grad_verts, None, grad_tex, None


########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, img_size=256):
        super(NeuralRenderer, self).__init__()
        self.renderer = NMR()

        # Adjust the core renderer
        self.renderer.renderer.image_size = img_size
        self.renderer.renderer.perspective = False

        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.renderer.eye = [0, 0, -2.732]

        # Make it a bit brighter for vis
        self.renderer.renderer.light_intensity_ambient = 0.8

        self.renderer.to_gpu()

        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.
        self.rr = Render(self.renderer)

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.renderer.light_intensity_ambient = 1
        self.renderer.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        self.renderer.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, cams, textures=None):
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        if textures is not None:
            return self.rr(verts, faces, textures, self.renderer)
        else:
            return self.rr(verts, faces, None, self.renderer)


########################################################################
############################## Tests ###################################
########################################################################
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

def teapot_deform_test():
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    img_file = 'birds3d/external/neural_renderer/examples/data/example2_ref.png'
    img_save_dir = 'birds3d/cachedir/nmr/'

    vertices, faces = neural_renderer.load_obj(obj_file)

    image_ref = scipy.misc.imread(img_file).astype('float32').mean(-1) / 255.
    image_ref = torch.autograd.Variable(torch.Tensor(image_ref[None, :, :]).cuda(device=0))

    mask_renderer = NeuralRenderer()
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
    cams = np.array([1., 0, 0, 1, 0, 0, 0], dtype=np.float32)
    cams_var = torch.autograd.Variable(torch.from_numpy(cams[None, :]).cuda(device=0))

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
