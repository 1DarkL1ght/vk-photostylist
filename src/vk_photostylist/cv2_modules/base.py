from abc import ABC, abstractmethod

import cv2

USE_OPENCL = cv2.ocl.haveOpenCL()

if USE_OPENCL:
    cv2.ocl.setUseOpenCL(True)


class BaseModule(ABC):
    @abstractmethod
    def __call__(self, image):
        if type(self) is BaseModule:
            raise NotImplementedError(
                "__call__ is not implemented for BaseModule class"
            )
        if cv2.ocl.useOpenCL():
            image = cv2.UMat(image)
