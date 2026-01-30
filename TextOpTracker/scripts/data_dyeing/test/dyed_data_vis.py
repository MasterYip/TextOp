"""
Visualization script for dyed motion data.

This script visualizes motion data that has been encoded with CLIP latent vectors:
1. Real-time motion visualization with predicted text labels
2. t-SNE trajectory visualization showing motion path in semantic space

Usage:
    python dyed_data_vis.py --zarr_path <path> --checkpoint <checkpoint>
    python dyed_data_vis.py --zarr_path ../../artifacts/g1_multimotion_noise_median/motion_dyed.zarr \
                             --checkpoint ../../../../MotionCLIP/exps/g1-model-xyz/checkpoint_0100.pth.tar
"""

import sys
import os
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent
DATA_DYEING_DIR = SCRIPT_DIR.parent
TEXTOP_TRACKER_DIR = DATA_DYEING_DIR.parent.parent
MOTIONCLIP_DIR = TEXTOP_TRACKER_DIR.parent / "MotionCLIP"

sys.path.insert(0, str(MOTIONCLIP_DIR))
sys.path.insert(0, str(DATA_DYEING_DIR))

import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
import yaml

# Import MotionCLIP as package
import motionclip
from motionclip import get_motion_clip, get_motion_text_mapping, retrieve_motions, encode_motions, get_datasets, clip

# Import local utilities
from diffusion_policy.utils.replay_buffer import ReplayBuffer
from visualize_utils import G1MotionVisualizer
from body_index_mapping import remap_isaaclab_to_motionclip


def load_vocabulary_categories(yaml_path=None):
    """Load vocabulary categories for text predictions."""
    if yaml_path is None:
        yaml_path = SCRIPT_DIR / "categories.yaml"
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten vocabulary into a list
    vocabulary = []
    for category, texts in config.get('vocabulary_categories', {}).items():
        vocabulary.extend(texts)
    # Remove duplicates
    vocabulary = list(set(vocabulary))

    return vocabulary


def load_motion_categories(yaml_path=None):
    """Load motion categories for t-SNE reference points."""
    if yaml_path is None:
        yaml_path = SCRIPT_DIR / "categories.yaml"
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('motion_categories', {})


class DyedDataVisualizer:
    """
    Visualizer for dyed motion data with CLIP latents.
    """
    
    def __init__(self, zarr_path, model, cfg, datasets, device='cuda'):
        """
        Initialize visualizer.
        
        Args:
            zarr_path: Path to dyed zarr dataset
            model: MotionCLIP model
            cfg: Model configuration
            datasets: Dataset collection
            device: Torch device
        """
        self.zarr_path = zarr_path
        self.model = model
        self.cfg = cfg
        self.datasets = datasets
        self.device = device
        
        # Load dyed dataset
        print(f"Loading dyed dataset from {zarr_path}")
        self.buffer = ReplayBuffer.copy_from_path(zarr_path)
        
        # Check for motion_latent field
        if 'motion_latent' not in self.buffer.data:
            raise ValueError("Dataset does not contain 'motion_latent' field. Please run data dyeing first.")
        
        print(f"Dataset loaded: {self.buffer.n_episodes} episodes, {self.buffer.n_steps} total frames")
        print(f"Fields: {list(self.buffer.data.keys())}")
        
        # Load vocabulary for text prediction
        self.vocabulary = load_vocabulary_categories()
        print(f"Loaded vocabulary: {len(self.vocabulary)} text descriptions")
        
        # Encode vocabulary with CLIP
        print("Encoding vocabulary with CLIP...")
        text_tokens = clip.tokenize(self.vocabulary).to(device)
        with torch.no_grad():
            self.text_features = self.model.clip_model.encode_text(text_tokens).float()
            self.text_features_norm = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        # Initialize G1 visualizer (auto-remap from IsaacLab to MotionCLIP order)
        self.motion_viz = G1MotionVisualizer(auto_remap=True)
    
    def predict_text_from_latent(self, motion_latent, top_k=5):
        """
        Predict text descriptions from motion latent vector.
        
        Args:
            motion_latent: Motion latent vector [latent_dim] or [batch, latent_dim]
            top_k: Number of top predictions to return
        
        Returns:
            List of (text, confidence) tuples
        """
        # Ensure motion_latent is 2D
        if motion_latent.dim() == 1:
            motion_latent = motion_latent.unsqueeze(0)
        
        # Normalize motion latent
        motion_latent_norm = motion_latent / motion_latent.norm(dim=-1, keepdim=True)
        
        # Compute similarity with all text descriptions
        with torch.no_grad():
            similarity = (100.0 * motion_latent_norm @ self.text_features_norm.t()).softmax(dim=-1)
        
        # Get top-k predictions
        values, indices = similarity[0].topk(top_k)
        
        predictions = []
        for idx, score in zip(indices, values):
            text = self.vocabulary[idx.item()]
            confidence = score.item() * 100
            predictions.append((text, confidence))
        
        return predictions
    
    def visualize_episode_with_text(self, episode_idx=0, output_path=None, fps=30):
        """
        Visualize an episode with real-time predicted text overlay.
        
        Args:
            episode_idx: Episode index to visualize
            output_path: Output video path (if None, uses default)
            fps: Frames per second
        """
        print(f"\n{'='*80}")
        print(f"Visualizing Episode {episode_idx} with Real-Time Text Prediction")
        print(f"{'='*80}\n")
        
        # Get episode data
        episode_data = self.buffer.get_episode(episode_idx)
        
        # Extract data
        body_pos = episode_data['body_pos']  # [T, 30, 3]
        motion_latents = episode_data['motion_latent']  # [T, latent_dim]
        
        T = body_pos.shape[0]
        print(f"Episode length: {T} frames ({T/fps:.2f} seconds)")
        
        # Set output path
        if output_path is None:
            output_path = SCRIPT_DIR / f"episode_{episode_idx}_realtime.gif"
        
        # Create figure with motion + text
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 1, height_ratios=[3, 1, 1], figure=fig)
        
        ax_3d = fig.add_subplot(gs[0], projection='3d')
        ax_text = fig.add_subplot(gs[1])
        ax_conf = fig.add_subplot(gs[2])
        
        # Setup 3D axes
        ax_3d.set_xlim(-0.7, 0.7)
        ax_3d.set_ylim(-0.7, 0.7)
        ax_3d.set_zlim(-0.7, 0.7)
        ax_3d.set_xticklabels([])
        ax_3d.set_yticklabels([])
        ax_3d.set_zticklabels([])
        ax_3d.view_init(azim=90, elev=0)
        ax_3d.set_title(f"Episode {episode_idx} - G1 Motion")
        
        # Setup text axes
        ax_text.axis('off')
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        
        # Setup confidence axes
        ax_conf.set_xlim(0, T)
        ax_conf.set_ylim(0, 100)
        ax_conf.set_xlabel('Frame')
        ax_conf.set_ylabel('Confidence (%)')
        ax_conf.set_title('Top-1 Prediction Confidence')
        ax_conf.grid(True, alpha=0.3)
        
        # Pre-compute all predictions
        print("Pre-computing text predictions...")
        all_predictions = []
        confidences_top1 = []
        confidences_top2 = []
        confidences_top3 = []
        for t in tqdm(range(T)):
            latent = torch.from_numpy(motion_latents[t]).float().to(self.device)
            predictions = self.predict_text_from_latent(latent, top_k=3)
            all_predictions.append(predictions)
            confidences_top1.append(predictions[0][1])  # Top-1 confidence
            confidences_top2.append(predictions[1][1])  # Top-2 confidence
            confidences_top3.append(predictions[2][1])  # Top-3 confidence
        
        # Animation update function
        def update(frame):
            # Clear axes
            ax_3d.clear()
            ax_text.clear()
            ax_conf.clear()
            
            # Setup 3D axes again
            ax_3d.set_xlim(-0.7, 0.7)
            ax_3d.set_ylim(-0.7, 0.7)
            ax_3d.set_zlim(-0.7, 0.7)
            ax_3d.set_xticklabels([])
            ax_3d.set_yticklabels([])
            ax_3d.set_zticklabels([])
            ax_3d.view_init(azim=90, elev=0)
            ax_3d.set_title(f"Episode {episode_idx} - Frame {frame}/{T}")
            
            # Draw G1 motion
            self.motion_viz.draw_frame(ax_3d, body_pos, frame)
            
            # Display predictions
            ax_text.axis('off')
            ax_text.set_xlim(0, 1)
            ax_text.set_ylim(0, 1)
            
            predictions = all_predictions[frame]
            text_y = 0.8
            ax_text.text(0.5, 0.95, 'Predicted Motion:', ha='center', va='top',
                        fontsize=14, fontweight='bold')
            
            for rank, (text, conf) in enumerate(predictions, 1):
                color = 'green' if rank == 1 else 'gray'
                alpha = 1.0 if rank == 1 else 0.6
                fontsize = 12 if rank == 1 else 10
                ax_text.text(0.5, text_y, f"{rank}. {text} ({conf:.1f}%)",
                           ha='center', va='top', fontsize=fontsize,
                           color=color, alpha=alpha)
                text_y -= 0.25
            
            # Plot confidence history
            ax_conf.set_xlim(0, T)
            ax_conf.set_ylim(0, 100)
            ax_conf.set_xlabel('Frame')
            ax_conf.set_ylabel('Confidence (%)')
            ax_conf.set_title('Top-3 Prediction Confidence')
            ax_conf.grid(True, alpha=0.3)
            
            # Plot top-3 confidence curves up to current frame
            ax_conf.plot(range(frame + 1), confidences_top1[:frame + 1], 'g-', linewidth=2.5, label='Top-1', alpha=0.9)
            ax_conf.plot(range(frame + 1), confidences_top2[:frame + 1], 'b-', linewidth=2, label='Top-2', alpha=0.7)
            ax_conf.plot(range(frame + 1), confidences_top3[:frame + 1], 'orange', linewidth=1.5, label='Top-3', alpha=0.6)
            ax_conf.axvline(frame, color='red', linestyle='--', alpha=0.5)
            if frame == 0:  # Add legend only on first frame
                ax_conf.legend(loc='upper right', fontsize=9)
            
            return ax_3d, ax_text, ax_conf
        
        # Create animation
        print(f"Creating animation...")
        ani = animation.FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)
        
        # Save
        print(f"Saving to {output_path}...")
        ani.save(output_path, writer='pillow', fps=fps)
        plt.close()
        
        print(f"✓ Visualization saved to: {output_path}")
        print(f"{'='*80}\n")
    
    def save_motion_embeddings(self, episode_idx=0, output_path=None, interval=1, format='txt'):
        """
        Save motion embeddings from an episode to file.
        
        Args:
            episode_idx: Episode index
            output_path: Output file path (if None, uses default)
            interval: Frame interval for sampling embeddings (1 = every frame, 10 = every 10th frame, etc.)
            format: Output format ('txt', 'npy', or 'pt')
        """
        print(f"\n{'='*80}")
        print(f"Saving Motion Embeddings from Episode {episode_idx}")
        print(f"{'='*80}\n")
        
        # Get episode motion latents
        episode_data = self.buffer.get_episode(episode_idx)
        motion_latents = episode_data['motion_latent']  # [T, latent_dim]
        T = motion_latents.shape[0]
        
        # Sample at interval
        sampled_latents = motion_latents[::interval]
        num_sampled = sampled_latents.shape[0]
        
        print(f"Episode length: {T} frames")
        print(f"Sampling interval: {interval}")
        print(f"Sampled embeddings: {num_sampled}")
        print(f"Embedding dimension: {sampled_latents.shape[1]}")
        
        # Set output path
        if output_path is None:
            output_path = SCRIPT_DIR / f"episode_{episode_idx}_embeddings_interval{interval}.{format}"
        
        # Save in requested format
        if format == 'txt':
            # Save as text file with one embedding per line
            np.savetxt(output_path, sampled_latents, fmt='%.8f', 
                      header=f"Episode {episode_idx} motion embeddings (interval={interval})\n"
                             f"Shape: {sampled_latents.shape}\n"
                             f"Each row is one embedding vector",
                      comments='# ')
            print(f"✓ Saved as text file (space-separated floats)")
        
        elif format == 'npy':
            # Save as numpy binary
            np.save(output_path, sampled_latents)
            print(f"✓ Saved as numpy binary")
        
        elif format == 'pt':
            # Save as PyTorch tensor
            torch.save(torch.from_numpy(sampled_latents), output_path)
            print(f"✓ Saved as PyTorch tensor")
        
        else:
            raise ValueError(f"Unknown format: {format}. Use 'txt', 'npy', or 'pt'")
        
        print(f"✓ Motion embeddings saved to: {output_path}")
        
        # Also save metadata
        meta_path = output_path.with_suffix('.meta.txt')
        with open(meta_path, 'w') as f:
            f.write(f"Episode: {episode_idx}\n")
            f.write(f"Total frames: {T}\n")
            f.write(f"Sampling interval: {interval}\n")
            f.write(f"Sampled frames: {num_sampled}\n")
            f.write(f"Embedding dimension: {sampled_latents.shape[1]}\n")
            f.write(f"Embedding shape: {sampled_latents.shape}\n")
            f.write(f"\nFrame indices (0-indexed):\n")
            for i, frame_idx in enumerate(range(0, T, interval)):
                f.write(f"  Embedding {i}: Frame {frame_idx}\n")
        
        print(f"✓ Metadata saved to: {meta_path}")
        print(f"{'='*80}\n")
        
        return output_path
    
    def visualize_tsne_trajectory(self, episode_idx=0, output_path=None, method='umap', n_neighbors=15, min_dist=0.1):
        """
        Visualize episode trajectory in low-dimensional semantic space.
        
        Args:
            episode_idx: Episode index to visualize
            output_path: Output image path (if None, uses default)
            method: Dimensionality reduction method ('umap' or 'tsne')
            n_neighbors: UMAP parameter - number of neighbors (default: 15)
            min_dist: UMAP parameter - minimum distance (default: 0.1)
        """
        print(f"\n{'='*80}")
        print(f"Visualizing Episode {episode_idx} {method.upper()} Trajectory")
        print(f"{'='*80}\n")
        
        # Get episode motion latents
        episode_data = self.buffer.get_episode(episode_idx)
        episode_latents = episode_data['motion_latent']  # [T, latent_dim]
        T = episode_latents.shape[0]
        
        # Load reference motions from dataset
        print("Loading reference motions for t-SNE...")
        motion_categories = load_motion_categories()
        motion_collection = get_motion_text_mapping(self.datasets)
        
        # Collect reference motions
        ref_labels = []
        ref_colors_map = {}
        motion_texts_to_retrieve = []
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(motion_categories)))
        
        for idx, (category, motion_texts) in enumerate(motion_categories.items()):
            ref_colors_map[category] = colors[idx]
            for motion_text in motion_texts:
                if motion_text in motion_collection:
                    motion_texts_to_retrieve.append(motion_text)
                    ref_labels.append(category)
        
        print(f"Retrieved {len(motion_texts_to_retrieve)} reference motions from {len(motion_categories)} categories")
        
        # Encode reference motions
        if len(motion_texts_to_retrieve) > 0:
            print("Encoding reference motions...")
            ref_motions = retrieve_motions(self.datasets, motion_collection, 
                                          motion_texts_to_retrieve, self.device)
            
            with torch.no_grad():
                ref_latents = encode_motions(self.model, ref_motions, self.device)
                ref_latents = ref_latents.cpu().numpy()
        else:
            print("Warning: No reference motions found")
            ref_latents = np.zeros((0, episode_latents.shape[1]))
        
        # Perform dimensionality reduction
        if method.lower() == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP is not installed. Install with: pip install umap-learn")
            
            print(f"Fitting UMAP on {len(ref_latents)} reference motions...")
            print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            # Fit UMAP only on reference motions
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                metric='cosine'
            )
            ref_tsne = reducer.fit_transform(ref_latents)
            
            # Project episode trajectory into the fitted space
            print(f"Projecting {len(episode_latents)} episode frames into fitted space...")
            episode_tsne = reducer.transform(episode_latents)
            
        elif method.lower() == 'tsne':
            print("Note: t-SNE doesn't support separate fitting and projection.")
            
            # Combine episode and reference latents
            all_latents = np.vstack([ref_latents, episode_latents])
            
            # Perform t-SNE
            perplexity = min(30, max(5, len(all_latents) // 3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            all_tsne = tsne.fit_transform(all_latents)
            
            # Split back into reference and episode
            ref_tsne = all_tsne[:len(ref_latents)]
            episode_tsne = all_tsne[len(ref_latents):]
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot reference points by category
        for category in motion_categories.keys():
            mask = np.array(ref_labels) == category
            if np.any(mask):
                ax.scatter(ref_tsne[mask, 0], ref_tsne[mask, 1],
                          c=[ref_colors_map[category]], label=category,
                          s=100, alpha=0.5, edgecolors='black', linewidth=0.5)
        
        # Plot episode trajectory
        # Color trajectory by time (progression)
        trajectory_colors = plt.cm.viridis(np.linspace(0, 1, T))
        
        # Draw trajectory line
        ax.plot(episode_tsne[:, 0], episode_tsne[:, 1], 
               'r-', linewidth=2, alpha=0.7, label='Episode Trajectory')
        
        # Draw trajectory points colored by time
        scatter = ax.scatter(episode_tsne[:, 0], episode_tsne[:, 1],
                           c=np.arange(T), cmap='viridis',
                           s=50, alpha=0.8, edgecolors='red', linewidth=1,
                           label='Episode Frames')
        
        # Mark start and end
        ax.scatter(episode_tsne[0, 0], episode_tsne[0, 1],
                  c='lime', s=300, marker='*', edgecolors='black', linewidth=2,
                  label='Start', zorder=10)
        ax.scatter(episode_tsne[-1, 0], episode_tsne[-1, 1],
                  c='red', s=300, marker='X', edgecolors='black', linewidth=2,
                  label='End', zorder=10)
        
        # Add colorbar for time progression
        cbar = plt.colorbar(scatter, ax=ax, label='Frame Number')
        
        ax.set_title(f'Episode {episode_idx} Trajectory in {method.upper()} CLIP Latent Space', fontsize=16)
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = SCRIPT_DIR / f"episode_{episode_idx}_{method}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {method.upper()} visualization saved to: {output_path}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize dyed motion data')
    parser.add_argument('--zarr_path', type=str, required=True,
                       help='Path to dyed zarr dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to MotionCLIP checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML (optional)')
    parser.add_argument('--episode', type=int, nargs='+', default=[100],
                       help='Episode index(es) to visualize. Can specify multiple: --episode 0 1 2 (default: [100])')
    parser.add_argument('--episodes-range', type=int, nargs=2, metavar=('START', 'END'),
                       help='Range of episodes to visualize (e.g., --episodes-range 0 10 for episodes 0-9)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--fps', type=int, default=30,
                       help='FPS for video output (default: 30)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: current script dir)')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save motion embeddings to file')
    parser.add_argument('--embedding-interval', type=int, default=1,
                       help='Frame interval for sampling embeddings (default: 1 = every frame)')
    parser.add_argument('--embedding-format', type=str, default='txt', choices=['txt', 'npy', 'pt'],
                       help='Format for saved embeddings (default: txt)')
    parser.add_argument('--only-embeddings', action='store_true',
                       help='Only save embeddings, skip visualizations')
    parser.add_argument('--dim-reduction', type=str, default='tsne', choices=['umap', 'tsne'],
                       help='Dimensionality reduction method for trajectory visualization (default: umap)')
    parser.add_argument('--umap-neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                       help='UMAP min_dist parameter (default: 0.1)')
    
    args = parser.parse_args()
    
    # Determine which episodes to visualize
    if args.episodes_range:
        episodes = list(range(args.episodes_range[0], args.episodes_range[1]))
        print(f"Visualizing episodes {args.episodes_range[0]} to {args.episodes_range[1]-1} ({len(episodes)} total)")
    else:
        episodes = args.episode if isinstance(args.episode, list) else [args.episode]
        print(f"Visualizing {len(episodes)} episode(s): {episodes}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = SCRIPT_DIR
    
    # Load MotionCLIP model
    print("Loading MotionCLIP model...")
    model, cfg = get_motion_clip(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Load datasets for motion retrieval
    print("Loading datasets...")
    
    # Get config path
    if args.config:
        config_path = args.config
    else:
        config_path = Path(args.checkpoint).parent / 'opt.yaml'
    
    if config_path.exists():
        from omegaconf import OmegaConf
        dataset_params = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    else:
        dataset_params = {
            'datapath': str(MOTIONCLIP_DIR / 'data/amass_db/amass_30fps_db.pt'),
            'dataset': 'amass',
            'num_frames': cfg.model.num_frames,
            'pose_rep': cfg.model.pose_rep,
            'glob': cfg.model.glob,
            'glob_rot': cfg.model.glob_rot,
            'translation': cfg.model.translation,
            'jointstype': cfg.model.jointstype,
            'vertstrans': cfg.model.vertstrans,
        }
    
    dataset_params['device'] = args.device
    dataset_params['datapath'] = str(MOTIONCLIP_DIR / 'data/g1_amass_db/amass_30fps_db.pt')
    _, clip_preprocess = clip.load("ViT-B/32", device=args.device, jit=False)
    datasets = get_datasets(dataset_params, clip_preprocess, split='all')
    
    # Create visualizer
    visualizer = DyedDataVisualizer(
        args.zarr_path,
        model,
        cfg,
        datasets,
        device=args.device
    )
    
    # Run visualizations
    print("\n" + "="*80)
    print("DYED DATA VISUALIZATION")
    print("="*80)

    # Validate episode indices
    max_episode_idx = visualizer.buffer.n_episodes - 1
    invalid_episodes = [ep for ep in episodes if ep > max_episode_idx]
    if invalid_episodes:
        print(f"\n[WARNING] Invalid episode indices (max={max_episode_idx}): {invalid_episodes}")
        episodes = [ep for ep in episodes if ep <= max_episode_idx]
        print(f"[WARNING] Continuing with valid episodes: {episodes}")
    
    if not episodes:
        print("\n[ERROR] No valid episodes to visualize!")
        return
    
    # Run visualizations for all episodes
    print("\n" + "="*80)
    print(f"DYED DATA VISUALIZATION - {len(episodes)} EPISODE(S)")
    print("="*80)
    
    all_videos = []
    all_tsne = []
    all_embeddings = []
    
    for i, episode_idx in enumerate(episodes, 1):
        print(f"\n[{i}/{len(episodes)}] Processing Episode {episode_idx}...")
        
        # Save embeddings if requested
        if args.save_embeddings or args.only_embeddings:
            embedding_path = output_dir / f"episode_{episode_idx}_embeddings_interval{args.embedding_interval}.{args.embedding_format}"
            visualizer.save_motion_embeddings(
                episode_idx=episode_idx,
                output_path=embedding_path,
                interval=args.embedding_interval,
                format=args.embedding_format
            )
            all_embeddings.append(embedding_path)
        
        # Skip visualizations if only saving embeddings
        if args.only_embeddings:
            continue
        
        # 1. Real-time text prediction visualization
        video_path = output_dir / f"episode_{episode_idx}_realtime.gif"
        visualizer.visualize_episode_with_text(
            episode_idx=episode_idx,
            output_path=video_path,
            fps=args.fps
        )
        all_videos.append(video_path)
        
        # 2. Dimensionality reduction trajectory visualization
        dim_red_path = output_dir / f"episode_{episode_idx}_{args.dim_reduction}.png"
        visualizer.visualize_tsne_trajectory(
            episode_idx=episode_idx,
            output_path=dim_red_path,
            method=args.dim_reduction,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist
        )
        all_tsne.append(dim_red_path)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    
    if args.only_embeddings:
        print(f"\nSaved {len(episodes)} episode embeddings:")
        for embedding_path in all_embeddings:
            print(f"  {embedding_path}")
    else:
        print(f"\nGenerated {len(episodes)} episode visualizations:")
        for idx in range(len(episodes)):
            print(f"\n  Episode {episodes[idx]}:")
            if all_videos:
                print(f"    Video: {all_videos[idx]}")
            if all_tsne:
                print(f"    {args.dim_reduction.upper()}: {all_tsne[idx]}")
            if all_embeddings:
                print(f"    Embeddings: {all_embeddings[idx]}")


if __name__ == '__main__':
    main()
