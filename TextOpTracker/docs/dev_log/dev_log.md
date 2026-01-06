### 20250105 data_dyeing

#file:data_collection.py  collects data from isaaclab.
In motionclip #file:minimal_load.py , you can use get_motion_clip to load motionclip model for data dyeing.
I want implement data dyeing to label the collected data with CLIP space latent vector in TextOpTracker/scripts/data_dyeing.
Requirements:
1. Implement motion format convertor to prepare motion before put into motion clip. You can look for the data format in  #sym:_load , there are several pose_rep options. Besides, note that for each frame's motion latent, you should sample around this frame to get the motion sequence as model input. About g1 AMASS dataset see #file:amass.py 
2. motion encoding to clip space. You can see how to use the motionclip encoder in #file:minimal_load.py . You should dye the motions in batch, adding the new motion latent to the dataset dict, and finally save it again.

### 20250106 DyedData Visualization

I hope to visualize the dyed data in such way in #file:dyed_data_vis.py  (if need more modules, put them in TextOpTracker/scripts/data_dyeing/test ) :
1. load the dyed zarr dataset, sample a episode data (motion sequence), then visualize the g1 data using the same way like #file:motion2text.py . The different is, I hope you display the most likely motion (and confidence) in realtime during the motion. About how to get the most likely motion text see #file:visualize.py  and #file:motion2text.py 
2. Prepare motion samples like #file:motionclip_tsne.py , and do tSNE for the samples latent to construct a dim-reducted space. Then we can project the motion_latent of the dyed data into this space, and the motion sequence may form a trajectory in the dim-reducted space, I hope you draw it out (with other motion sample points to reflect the semantic of the zone)
3. you can import motionclip as a package to call the needed functions, note to expose the entry in #file:__init__.py 