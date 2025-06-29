### Troubleshooting Multiscale Deformable Attention Installation (MSDA)
Code updates I had to make (should already be implemented):
* change `type()` to `scaler_type()`

Additional steps I had to follow to install MSDA:
* Updated Windows GPU drivers through the NVIDIA App (previously called GeForce Experience) and restarted my PC, then installed/upgraded cuda toolkit through the runfile option
  * Steps followed through [this video](https://www.youtube.com/watch?v=JaHVsZa2jTc&ab_channel=NVIDIADeveloper)
  * cuda toolkit [install](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local) (use runfile option)
  * According to the video, Windows drivers automatically apply to WSL now
