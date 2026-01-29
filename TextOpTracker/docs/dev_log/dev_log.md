
## BUG

- 20260106 DataCollection Episode is not continuous

## TODO

- 20260106 Double Check DataDyeing Script

## Prompt

### 20260129 data_collection motion id

for now the `deterministic` mode in #file:data_collection.py has a problem: the collected motions are not in order (motions are not arranged like motion1:eps0-49, motion2:eps50-99, etc), hence you don't know what the motion is the sample collected from.

Therefore, I hope you add "motion_idx" of the current env in #sym:extract_robot_state  (You can get it from #file:commands_collection.py  ) function, so that in #file:data_collection.py ,  we can add motion id to the dataset of each episode.

This is convenient for later on motion dyeing.

### 20260108 data_collection improvement

I need to implement a new data collection strategy. Requirements:
1. Basic logic. For example, input M motion files, in order to ensure action distribution coverage, for each motion sequence sample N times (full length). There are totally M*N episodes saved. Others keep as the origin version.
2. Implement #file:commands_collection.py . It automatically assign the motion reference to fulfill the N*M sampling task, If the motion are called to resample in case of terminated (task failed) this do not counted as completed, If the tasks are all assigned, the reset the idling env to a default pos to avoid env termination at each frame. You can refer to #file:commands_multi.py for the origin file.
3.  Update #file:data_collection.py #file:collect_dataset.sh  to support this collect mode, and keep backward compatibility. Update #file:data_collection.yaml for new configs for this collection mode. Update #file:README.md for the new mode instruction.

### 20260108 data_selection

I hope to implement #file:data_selection.py . Requirement:
1. #file:data_selection.py  replays motions like #file:replay_npz_multi.py for user, and user can input their wanted motion env_id in terminal one by one. Due to there is too much motions, you should  provide command to change motions (e.g. 1000 in total, vis 100 each time, provide a command to turn "page". in each  page, env_id range form 0-99) When done, it copies the selected motion to the new specified folder. Override existing same named motion, and save  the motion list in yaml.

### 20250105 data_dyeing

#file:data_collection.py  collects data from isaaclab.
In motionclip #file:minimal_load.py , you can use get_motion_clip to load motionclip model for data dyeing.
I want implement data dyeing to label the collected data with CLIP space latent vector in TextOpTracker/scripts/data_dyeing.
Requirements:
1. Implement motion format convertor to prepare motion before put into motion clip. You can look for the data format in  #sym:_load , there are several pose_rep options. Besides, note that for each frame's motion latent, you should sample around this frame to get the motion sequence as model input. About g1 AMASS dataset see #file:amass.py 
2. motion encoding to clip space. You can see how to use the motionclip encoder in #file:minimal_load.py . You should dye the motions in batch, adding the new motion latent to the dataset dict, and finally save it again.

**text embedding calculation**
I found this gap is indeed innegligible. I hope you create a new motion dyeing method in #file:data_dyeing.py . This loads a vast vocabulary from ymal file, calculate confidence of the volcab, then compute the volcab confidence weighted  embedding as the embedding. In this way, the embedding aligns with the text embedding. Refer to #file:dyed_data_vis.py  for the detail of text similarity computation

### 20250106 DyedData Visualization

I hope to visualize the dyed data in such way in #file:dyed_data_vis.py  (if need more modules, put them in TextOpTracker/scripts/data_dyeing/test ) :
1. load the dyed zarr dataset, sample a episode data (motion sequence), then visualize the g1 data using the same way like #file:motion2text.py . The different is, I hope you display the most likely motion (and confidence) in realtime during the motion. About how to get the most likely motion text see #file:visualize.py  and #file:motion2text.py 
2. Prepare motion samples like #file:motionclip_tsne.py , and do tSNE for the samples latent to construct a dim-reducted space. Then we can project the motion_latent of the dyed data into this space, and the motion sequence may form a trajectory in the dim-reducted space, I hope you draw it out (with other motion sample points to reflect the semantic of the zone)
3. you can import motionclip as a package to call the needed functions, note to expose the entry in #file:__init__.py 

Fix bug:
I notice a important bug: the body position ordering is different between #file:data_collection.py  and motion clip. So for #file:data_dyeing.py  and #file:dyed_data_vis.py , They all need to remap body indice and joint indice before putting into motion clip.

For MotionCLIP side, the order can be found in #file:g1_amass_utils.py 
For TeleOP side, I think the order are put in alphabet order.

**save motion embedding**
A problem shows up: There is a semantic gap between text embedding & motion embedding. The cond co-diffuse is trained on motion embedding, not works good for text embedding.
I hope you add an option to save the motion embedding  with interval to txt in #file:dyed_data_vis.py . So that I can try whether the motion embedding can work for cond co-diffuse