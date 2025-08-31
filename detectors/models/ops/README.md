### Troubleshooting Multiscale Deformable Attention Installation (MSDA)

__When reinstalling MSDA you must delete `ops/build` and probably the `ops/dist` and `ops/MultiScaleDeformableAttention.egg-info` dirs__

Code updates I had to make (should already be implemented):
* in `ms_deform_attn_cuda.cu` on line `64` & `134` change `value.type()` to `value.scaler_type()`

Additional steps I had to follow to install MSDA:
* Updated Windows GPU drivers through the NVIDIA App (previously called GeForce Experience) and restarted my PC, then installed/upgraded cuda toolkit through the runfile option
  * Steps followed through [this video](https://www.youtube.com/watch?v=JaHVsZa2jTc&ab_channel=NVIDIADeveloper)
  * cuda toolkit [install](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local) (use runfile option)
  * According to the video, Windows drivers automatically apply to WSL now
* Point your `PATH` to the new cuda version by adding the following lines to `.bashrc` and then run `source .bashrc`
```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9:/lib64:$LD_LIBRARY_PATH
```
