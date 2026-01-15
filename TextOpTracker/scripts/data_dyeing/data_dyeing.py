"""
Data Dyeing Script for G1 Motion Dataset.

Encodes collected motion data into CLIP latent space using MotionCLIP.
Loads zarr dataset from data_collection.py, processes motions in batches,
and adds motion latent vectors to the dataset.

Usage:
    python data_dyeing.py
    python data_dyeing.py --config custom_config.yaml
"""

import sys
import os
from pathlib import Path

# Add paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent
MOTIONCLIP_DIR = ROOT_DIR / "MotionCLIP"
sys.path.append(str(MOTIONCLIP_DIR))
sys.path.append(str(ROOT_DIR / "TextOpTracker"))

import torch
import numpy as np
import zarr
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import yaml
import clip
from scipy import interpolate

# Import MotionCLIP utilities
from src.utils.get_model_and_data import get_motion_clip

# Import local utilities
from motion_converter import G1MotionConverter, extract_motion_window, create_motion_batches
from replay_buffer import ReplayBuffer


class MotionDataDyer:
    """
    Encodes motion data into CLIP latent space.
    
    Processes collected G1 motion data and adds motion latent vectors
    using a trained MotionCLIP model.
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the data dyer.
        
        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.device = torch.device(cfg.motionclip.device)
        
        print("="*80)
        print("Motion Data Dyeing - CLIP Latent Encoding")
        print("="*80)
        
        # Load MotionCLIP model
        print(f"\n[1/4] Loading MotionCLIP model...")
        print(f"  Checkpoint: {cfg.motionclip.checkpoint_path}")
        
        if not os.path.exists(cfg.motionclip.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {cfg.motionclip.checkpoint_path}")
        
        self.model, self.model_cfg = get_motion_clip(
            checkpoint_path=cfg.motionclip.checkpoint_path,
            device=cfg.motionclip.device
        )
        self.model.eval()
        
        print(f"  Model loaded successfully!")
        print(f"  Latent dim: {self.model_cfg.model.latent_dim}")
        print(f"  Num frames: {self.model_cfg.model.num_frames}")
        
        # Initialize motion converter
        self.converter = G1MotionConverter(
            pose_rep=cfg.encoding.pose_rep,
            translation=cfg.encoding.translation,
            glob=cfg.encoding.glob,
            device=self.device
        )
        
        # Validate window size matches model
        if cfg.encoding.window_size != self.model_cfg.model.num_frames:
            print(f"\n  WARNING: window_size ({cfg.encoding.window_size}) != "
                  f"model.num_frames ({self.model_cfg.model.num_frames})")
            print(f"  Using model.num_frames = {self.model_cfg.model.num_frames}")
            self.window_size = self.model_cfg.model.num_frames
        else:
            self.window_size = cfg.encoding.window_size
        
        # Store FPS conversion parameters
        self.origin_fps = cfg.encoding.get('origin_fps', 30)
        self.target_fps = cfg.encoding.get('target_fps', 30)
        self.interp_method = cfg.encoding.get('interp_method', 'linear')
        
        if self.origin_fps != self.target_fps:
            print(f"\n  FPS Conversion enabled: {self.origin_fps} Hz → {self.target_fps} Hz")
            print(f"  Interpolation method: {self.interp_method}")
        
        # Load vocabulary if using text-aligned encoding
        self.text_features_norm = None
        if cfg.encoding.get('use_text_alignment', False):
            self._load_vocabulary()
    
    def _load_vocabulary(self):
        """Load vocabulary and encode with CLIP text encoder."""
        print(f"\n  Loading vocabulary for text alignment...")
        vocab_path = self.cfg.encoding.vocabulary_path
        
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        # Load vocabulary from YAML
        with open(vocab_path, 'r') as f:
            vocab_config = yaml.safe_load(f)
        
        # Flatten vocabulary into a list
        vocabulary = []
        for category, texts in vocab_config.get('vocabulary_categories', {}).items():
            vocabulary.extend(texts)
        # Remove duplicates
        vocabulary = list(set(vocabulary))

        print(f"  Loaded {len(vocabulary)} text descriptions")
        
        # Encode vocabulary with CLIP (same as dyed_data_vis.py)
        print(f"  Encoding vocabulary with CLIP text encoder...")
        text_tokens = clip.tokenize(vocabulary).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.clip_model.encode_text(text_tokens).float()
            self.text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        print(f"  Text features shape: {self.text_features_norm.shape}")
        print(f"  Text alignment enabled!")
    
    def load_dataset(self):
        """Load input zarr dataset."""
        print(f"\n[2/4] Loading input dataset...")
        print(f"  Path: {self.cfg.input.zarr_path}")
        
        if not os.path.exists(self.cfg.input.zarr_path):
            raise FileNotFoundError(f"Dataset not found: {self.cfg.input.zarr_path}")
        
        # Load using ReplayBuffer (same as data_collection.py)
        self.buffer = ReplayBuffer.copy_from_path(self.cfg.input.zarr_path)
        
        # Get dataset info
        self.total_frames = self.buffer.n_steps
        
        print(f"  Total frames: {self.total_frames}")
        print(f"  Total episodes: {self.buffer.n_episodes}")
        print(f"  Fields: {list(self.buffer.data.keys())}")
        
        # Pre-compute episode indices for each frame
        # This maps each frame to its episode index and boundaries
        episode_ends = self.buffer.episode_ends[:]
        self.frame_to_episode = np.zeros(self.total_frames, dtype=np.int64)
        self.episode_starts = np.zeros(len(episode_ends), dtype=np.int64)
        self.episode_ends_arr = episode_ends.copy()
        
        for ep_idx in range(len(episode_ends)):
            start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            end = episode_ends[ep_idx]
            self.episode_starts[ep_idx] = start
            self.frame_to_episode[start:end] = ep_idx
        
        print(f"  Episode boundaries computed")
        
        # Check required fields
        required_fields = ['body_pos', 'body_rot']
        for field in required_fields:
            if field not in self.buffer.data:
                raise ValueError(f"Required field '{field}' not found in dataset")
        
        print(f"  Dataset loaded successfully!")
    
    def encode_motions(self):
        """Encode all motions to CLIP latent space."""
        print(f"\n[3/4] Encoding motions to CLIP latent space...")
        print(f"  Window size: {self.window_size}")
        print(f"  Batch size: {self.cfg.encoding.batch_size}")
        print(f"  Boundary mode: {self.cfg.encoding.boundary_mode}")
        
        # Prepare output array for latents
        latent_dim = self.model_cfg.model.latent_dim
        all_latents = np.zeros((self.total_frames, latent_dim), dtype=np.float32)
        
        # Create batches
        batches = create_motion_batches(
            self.total_frames,
            self.window_size,
            self.cfg.encoding.batch_size,
            self.cfg.encoding.boundary_mode
        )
        
        print(f"  Total batches: {len(batches)}")
        
        # Process batches
        with torch.no_grad():
            pbar = tqdm(batches, desc="Encoding motions")
            
            for batch_idx, (batch_start, batch_end, center_indices) in enumerate(pbar):
                # Prepare batch data
                batch_motions = []
                batch_lengths = []
                
                for center_idx in center_indices:
                    # Extract motion window
                    window_data = self._extract_window(center_idx)
                    
                    # Convert to MotionCLIP format
                    motion_tensor = self.converter.convert(
                        body_pos=window_data['body_pos'],
                        body_rot=window_data['body_rot'],
                        body_lin_vel=window_data.get('body_lin_vel'),
                        body_ang_vel=window_data.get('body_ang_vel')
                    )
                    # Shape: [30 or 31, feat_dim, window_size]
                    
                    batch_motions.append(motion_tensor)
                    batch_lengths.append(self.window_size)
                
                # Stack batch
                batch_motions = torch.stack(batch_motions, dim=0)  # [B, bodies, feat_dim, T]
                batch_lengths = torch.tensor(batch_lengths, dtype=torch.long, device=self.device)
                
                # Encode with MotionCLIP
                mask = self.model.lengths_to_mask(batch_lengths)
                dummy_labels = torch.zeros(len(center_indices), dtype=torch.long, device=self.device)
                
                encoded = self.model.encoder({
                    'x': batch_motions,
                    'y': dummy_labels,
                    'mask': mask
                })
                
                latent_features = encoded['mu']  # [B, latent_dim]
                
                # Apply text alignment if enabled
                if self.text_features_norm is not None:
                    latent_features = self._align_to_text(latent_features)
                
                # Store latents
                all_latents[batch_start:batch_end] = latent_features.cpu().numpy()
                
                # Update progress
                if self.cfg.logging.verbose and batch_idx % self.cfg.logging.log_interval == 0:
                    pbar.set_postfix({
                        'batch': f"{batch_idx+1}/{len(batches)}",
                        'frames': f"{batch_end}/{self.total_frames}"
                    })
        
        print(f"  Encoding complete!")
        print(f"  Latent shape: {all_latents.shape}")
        
        return all_latents
    
    def _resample_motion_window(self, motion_data):
        """
        Resample motion window from origin_fps to target_fps.
        
        Always produces exactly self.window_size frames at target_fps.
        
        Args:
            motion_data: Dict with 'body_pos', 'body_rot', etc. [T_orig, ...]
        
        Returns:
            resampled_data: Dict with resampled motion data [window_size, ...]
        """
        if self.origin_fps == self.target_fps:
            return motion_data  # No resampling needed
        
        # Get original number of frames
        T_orig = motion_data['body_pos'].shape[0]
        
        # Target is exactly self.window_size frames
        T_new = self.window_size
        
        # Create time arrays
        # Original: frames at origin_fps
        t_orig = np.arange(T_orig) / self.origin_fps
        
        # Target: exactly window_size frames at target_fps, centered
        # This ensures we sample {..., t-1/target_fps, t, t+1/target_fps, ...}
        duration_target = (T_new - 1) / self.target_fps
        t_new = np.linspace(0, duration_target, T_new)
        
        # Resample each field
        resampled_data = {}
        
        for key, data in motion_data.items():
            if data is None:
                resampled_data[key] = None
                continue
            
            # Handle different interpolation methods
            if self.interp_method == 'nearest':
                kind = 'nearest'
            elif self.interp_method == 'cubic':
                kind = 'cubic'
            else:  # linear
                kind = 'linear'
            
            # For multi-dimensional data, interpolate along first axis
            original_shape = data.shape
            data_flat = data.reshape(T_orig, -1)  # [T, features]
            
            # Create interpolator
            f = interpolate.interp1d(t_orig, data_flat, axis=0, kind=kind, 
                                    fill_value='extrapolate')
            
            # Resample to exactly window_size frames
            resampled_flat = f(t_new)
            
            # Reshape back
            new_shape = (T_new,) + original_shape[1:]
            resampled_data[key] = resampled_flat.reshape(new_shape)
        
        return resampled_data
    
    def _align_to_text(self, motion_latents):
        """
        Align motion latents to text embedding space using vocabulary-weighted averaging.
        
        This computes similarity between motion latents and all text embeddings,
        then creates a new embedding as a weighted average of text embeddings.
        This bridges the semantic gap between motion and text embeddings.
        
        Args:
            motion_latents: Motion latent vectors [B, latent_dim]
        
        Returns:
            text_aligned_latents: Text-aligned latent vectors [B, latent_dim]
        """
        # Normalize motion latents (same as dyed_data_vis.py)
        motion_latents_norm = motion_latents / motion_latents.norm(dim=-1, keepdim=True)
        
        # Compute similarity with all text descriptions [B, num_texts]
        # Using same formula as dyed_data_vis.py: (100.0 * motion @ text.T).softmax()
        similarity = (100.0 * motion_latents_norm @ self.text_features_norm.t()).softmax(dim=-1)
        
        # Compute weighted average of text embeddings
        # text_aligned = sum(similarity[i] * text_embedding[i])
        # Shape: [B, num_texts] @ [num_texts, latent_dim] = [B, latent_dim]
        text_aligned_latents = similarity @ self.text_features_norm
        
        return text_aligned_latents
    
    def _extract_window(self, center_idx):
        """Extract motion window for a specific frame."""
        # Determine which episode this frame belongs to
        episode_idx = self.frame_to_episode[center_idx]
        episode_start = self.episode_starts[episode_idx]
        episode_end = self.episode_ends_arr[episode_idx]
        
        # Create data dict from buffer
        data_dict = {
            'body_pos': self.buffer['body_pos'][:],
            'body_rot': self.buffer['body_rot'][:],
        }
        
        # Add optional fields if available
        if 'body_lin_vel' in self.buffer.data:
            data_dict['body_lin_vel'] = self.buffer['body_lin_vel'][:]
        if 'body_ang_vel' in self.buffer.data:
            data_dict['body_ang_vel'] = self.buffer['body_ang_vel'][:]
        
        # Extract window at original FPS
        # Need to adjust window size for original FPS
        if self.origin_fps != self.target_fps:
            # Calculate window size at original FPS to get target duration
            target_duration = (self.window_size - 1) / self.target_fps  # Duration in seconds
            origin_window_size = int(target_duration * self.origin_fps) + 1
        else:
            origin_window_size = self.window_size
        
        window_data = extract_motion_window(
            data_dict,
            center_idx,
            origin_window_size,
            self.cfg.encoding.boundary_mode,
            episode_start=episode_start,
            episode_end=episode_end
        )
        
        # Resample to target FPS if needed
        if self.origin_fps != self.target_fps:
            window_data = self._resample_motion_window(window_data)
        
        return window_data
    
    def save_dataset(self, latents):
        """Save dataset with added latent vectors."""
        print(f"\n[4/4] Saving dataset with latent vectors...")
        print(f"  Output path: {self.cfg.output.zarr_path}")
        
        # Create output directory
        output_dir = os.path.dirname(self.cfg.output.zarr_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Add latent field to buffer's data
        latent_field = self.cfg.output.latent_field_name
        
        # Check if field exists
        if latent_field in self.buffer.data:
            if self.cfg.output.overwrite:
                print(f"  Overwriting existing '{latent_field}' field...")
                # Delete from buffer if using zarr backend
                if self.buffer.backend == 'zarr':
                    del self.buffer.data[latent_field]
            else:
                raise ValueError(f"Field '{latent_field}' already exists. Set overwrite=true to replace.")
        
        print(f"  Adding '{latent_field}' field...")
        
        # Add latent field to buffer's data dict
        if self.buffer.backend == 'numpy':
            self.buffer.data[latent_field] = latents
        else:
            # For zarr backend, we need to add it to the data group
            # Set compression
            if self.cfg.output.compressor == 'blosc':
                from numcodecs import Blosc
                compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
            elif self.cfg.output.compressor == 'default':
                compressor = ReplayBuffer.resolve_compressor('default')
            else:
                compressor = None
            
            self.buffer.data.array(
                latent_field,
                latents,
                chunks=(1000, latents.shape[1]),
                dtype=np.float32,
                compressor=compressor
            )
        
        # Save buffer to output path
        print(f"  Saving buffer to disk...")
        
        # Set compression for saving
        compressor_dict = {latent_field: self.cfg.output.compressor}
        if self.cfg.output.compressor == 'blosc':
            compressor_dict = {latent_field: 'disk'}  # Use 'disk' for high compression
        
        self.buffer.save_to_path(
            self.cfg.output.zarr_path,
            compressors=compressor_dict,
            if_exists='replace'
        )
        
        print(f"  Dataset saved successfully!")
        if os.path.exists(self.cfg.output.zarr_path):
            # Calculate directory size for zarr
            total_size = sum(f.stat().st_size for f in Path(self.cfg.output.zarr_path).rglob('*') if f.is_file())
            print(f"  Total size: {total_size / 1024**2:.2f} MB")
        
        # Print summary
        print("\n" + "="*80)
        print("Data Dyeing Complete!")
        print("="*80)
        print(f"Input dataset: {self.cfg.input.zarr_path}")
        print(f"Output dataset: {self.cfg.output.zarr_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"Total episodes: {self.buffer.n_episodes}")
        print(f"Latent field: '{latent_field}' with shape {latents.shape}")
        print(f"Latent dim: {latents.shape[1]}")
        print("="*80)


@hydra.main(version_base=None, config_path=".", config_name="data_dyeing")
def main(cfg: DictConfig):
    """Main entry point."""
    
    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize dyer
    dyer = MotionDataDyer(cfg)
    
    # Load dataset
    dyer.load_dataset()
    
    # Encode motions
    latents = dyer.encode_motions()
    
    # Save dataset
    dyer.save_dataset(latents)


if __name__ == '__main__':
    main()
