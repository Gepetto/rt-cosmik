"""Make smplfitter importable on torch < 2.5.

smplfitter 0.5 uses ``torch.nn.Buffer``, which arrived in torch 2.5. This
container is pinned to torch 2.4.1 because the TensorRT engines and the NLF
torchscript were built against it, so the shim is the cheap side of that trade.

Import this module *before* smplfitter. Verified in the upstream guide: with the
shim, ``.to("cuda")`` still moves buffers correctly and fitted results match.
"""
import torch
import torch.nn as nn

if not hasattr(nn, "Buffer"):

    class Buffer(torch.Tensor):
        def __new__(cls, data=None, *, persistent=True):
            if data is None:
                data = torch.empty(0)
            tensor = torch.Tensor._make_subclass(cls, data, data.requires_grad)
            tensor._persistent = persistent
            return tensor

    _original_setattr = nn.Module.__setattr__

    def _setattr(self, name, value):
        if isinstance(value, Buffer):
            plain = value.as_subclass(torch.Tensor)
            if "_buffers" in self.__dict__:
                for mapping in (self.__dict__, self.__dict__.get("_parameters", {})):
                    mapping.pop(name, None)
                self.register_buffer(name, plain,
                                     persistent=getattr(value, "_persistent", True))
                return
            value = plain
        _original_setattr(self, name, value)

    nn.Buffer = Buffer
    nn.Module.__setattr__ = _setattr
