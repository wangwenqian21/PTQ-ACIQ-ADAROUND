from utils.misc import Singleton
import abc

INFERENCE_ONLY = False

class QuantizationManagerBase(metaclass=Singleton):
    def __init__(self):
        pass

    def __enter__(self): # with QM 那步直接enable了。退出时再disable
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()

    @abc.abstractclassmethod
    def createTruncationManager(self, args, qparams):
        return

    def enable(self):
        if self.quantize:
            self.enabled = True if not self.disable_quantization else False
            self.op_manager.enable() # op_manager.enable这步将所有的nn.linear换成LinearWithId

    def disable(self):
        self.enabled = False
        self.op_manager.disable()


    def quantize_instant(self, tensor, tag="", quantize_tensor=False):
        if self.quantize and quantize_tensor:
            return self.op_manager.quantize_instant(tensor, tag)
        else:
            return tensor
