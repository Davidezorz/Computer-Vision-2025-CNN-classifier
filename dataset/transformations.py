
class ChannelMean:                                                              # tensor shape  C H W, but
    def __call__(self, tensor):                                                 # we mean on the channels, 
        return tensor.mean(dim=0, keepdim=True)                                 # then -> 1 H W
    


class Times255:                                                              
    def __call__(self, tensor):                                                 
        return  tensor * 255