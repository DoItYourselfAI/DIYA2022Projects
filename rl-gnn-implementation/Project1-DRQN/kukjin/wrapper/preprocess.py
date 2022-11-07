from gym.core import ObservationWrapper
from gym.spaces import Box

# from scipy.misc import imresize
import cv2

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self,env)
        
        self.img_size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def observation(self, img):
        """what happens to each observation"""
        
        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = img[34:-16, :, :]
        
        # resize image
        #print(f'Before img.shape 1: {img.shape}') # (160, 160, 3)
        img = cv2.resize(img, self.img_size)
        #print(f'Before img.shape 2: {img.shape}') # (84, 84, 3)
        img = img.mean(-1,keepdims=True)
        #print(f'After img.shape: {img.shape}') # (84, 84, 1)
        img = img.astype('float32') / 255.
              
        return img