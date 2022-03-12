#命名格式

#主網路_敘述+功能_敘述...

import webbrowser
from showTensorboard import showTensorboard
import os
  
names = {8000:'a', 
        8001:'backup/useGAN_E(ITCNET)-D(1)-D(ITCNET_3layer)_MSELoss_discriminator_MSELoss[6]_tripletx.0.03',
        8002:'useGAN_threshold1_E(ITCNET)-D(1)-D(ITCNET_3layer)_MSELoss_discriminator_MSELoss[0]',
        8003:'useGAN_threshold0.9_E(ITCNET)-D(1)-D(ITCNET_3layer)_MSELoss_discriminator_MSELoss[0]',
        8004:'useGAN_threshold0.5_E(ITCNET)-D(1)-D(ITCNET_3layer)_MSELoss_discriminator_MSELoss[0]',
        8005:'useGAN_origin_E(ITCNET)-D(1)-D(ITCNET_3layer)_MSELoss_discriminator_MSELoss[0]',
        8007:'ITCNET_E(ITCNET)-D(null)-D(ITCNET_3layer)_MSELoss_None_MSELoss_MSELossToTripletLoss_BCELoss[0]',
        8008: "useGAN_threshold0.5_E(ITCNET)-D(1)-D(ITCNET_3layer)_MSELoss_discriminator_MSELoss[0]"
        }

#0.00001_0.000001
ports = [ i for i in range(8000, 8001)] 
for p in ports:
    tensorboard = showTensorboard("./runs", names[p], p, showWeb = True).start()