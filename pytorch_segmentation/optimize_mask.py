import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
import tqdm


class SegOptimizerModel(torch.nn.Module):
    def __init__(self,model,image,seg,checkpoint='saved_psp/mask/12-14_21-03/best_model.pth'):
        super(SegOptimizerModel, self).__init__()
        self.model = model
        self.seg = torch.nn.Parameter(data=torch.tensor(seg),requires_grad=True)
        self.image = torch.nn.Parameter(data=torch.tensor(image),requires_grad=False)
        for p in self.model.parameters():
            p.requires_grad=False       
        params  = dict([('.'.join(k.split('.')[1:]),v) for k,v in torch.load(checkpoint)['state_dict'].items()])
        self.model.load_state_dict(params)

    def forward(self,data=None):
        data = torch.cat([self.image,self.seg],1)
        return self.model(data)
    
def optimize_mask(model,loss,device,image,seg):        
    mmodel = SegOptimizerModel(model,image,seg).to(device)
    lr = .1
    for i in tqdm.tqdm(range(1000)):
        output = mmodel(data)
        l = loss(output[0], target*0)
        l += loss(output[1], target*0) * 0.4
        output = output[0]
        l = l.mean()
        l.backward()
        with torch.no_grad():
            mmodel.seg-=mmodel.seg.grad*lr#+(torch.rand(*mmodel.seg.grad.size(),device=device)-.5)*lr*0.01
            mmodel.seg[:] = torch.clamp(mmodel.seg,0,1)
            mmodel.seg.grad[:] = 0
    return mmodel.seg.cpu().argmax(1).numpy()

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
 
def main(args):
    config = json.load(open(arg.config)
    loader = getattr(dataloaders, config['train_loader']['type'])(*args,return)_id=True, **config['train_loader']['args']) 
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])
    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    checkpoint=arg.checkpoint
    device = torch.device('cuda:0')

    for batch  in train_loader:
        data, target,_id = batch
        image = data[:,:1,:,:].numpy()
        seg = data[:,1:,:,:].numpy()                 
        mmodel = SegOptimizerModel(model,image,seg).to(device)
        seg = optimize_mask(model,loss,device,image,seg)
        for i in seg:
            cv2.imwrite(os.path.join(args.seve,str(id)+'.png'),(i*255/train_loader.dataset.num_classes).astype(np.uint8))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to the .pth model checkpoint')
    args = parser.parse_args()    
    main(args)