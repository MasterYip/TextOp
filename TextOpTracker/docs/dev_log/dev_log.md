### 20250105 data_dyeing

#file:data_collection.py  collects data from isaaclab.
In motionclip #file:minimal_load.py , you can use get_motion_clip to load motionclip model for data dyeing.
I want implement data dyeing to label the collected data with CLIP space latent vector in TextOpTracker/scripts/data_dyeing.
Requirements:
1. Implement motion format convertor to prepare motion before put into motion clip. You can look for the data format in  #sym:_load , there are several pose_rep options. Besides, note that for each frame's motion latent, you should sample around this frame to get the motion sequence as model input. About g1 AMASS dataset see #file:amass.py 
2. motion encoding to clip space. You can see how to use the motionclip encoder in #file:minimal_load.py . You should dye the motions in batch, adding the new motion latent to the dataset dict, and finally save it again.