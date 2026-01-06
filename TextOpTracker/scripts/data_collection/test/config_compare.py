"""
Script to compare configurations loaded from different sources:
1. Pickle files (env.pkl, agent.pkl) from checkpoint
2. Hydra registration (task name + entry point)

Usage:
    python config_compare.py \
        --pickle_path /path/to/checkpoint/params \
        --task_name Isaac-G1-Flat-ProjGravObs-v0 \
        --entry_point rsl_rl_cfg_entry_point
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Setup Isaac Lab environment
from isaaclab.app import AppLauncher

# Parse minimal args for AppLauncher
parser = argparse.ArgumentParser(description="Compare configs from pickle vs Hydra")
parser.add_argument("--pickle_path", type=str, required=True, help="Path to params directory containing env.pkl and agent.pkl")
parser.add_argument("--task_name", type=str, required=True, help="Task name for Hydra registration")
parser.add_argument("--entry_point", type=str, default="rsl_rl_cfg_entry_point", help="Entry point name")
parser.add_argument("--ignore_keys", type=str, nargs="*", default=[], help="Keys to ignore in comparison")
parser.add_argument("--output", type=str, default=None, help="Output file for comparison report")

args = parser.parse_args()

# Launch Isaac Sim
app_launcher_args = argparse.Namespace(
    headless=True,
    livestream=False,
    device="cpu",
    enable_cameras=False,
)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
from isaaclab.utils.io.pkl import load_pickle
from isaaclab_tasks.utils.hydra import register_task_to_hydra


class ConfigComparator:
    """Recursive configuration comparator."""
    
    def __init__(self, ignore_keys: List[str] = None):
        self.ignore_keys = set(ignore_keys or [])
        self.differences = []
        
    def compare(self, obj1: Any, obj2: Any, path: str = "root") -> List[Tuple[str, Any, Any]]:
        """
        Recursively compare two objects and return list of differences.
        
        Args:
            obj1: First object (e.g., from pickle)
            obj2: Second object (e.g., from Hydra)
            path: Current path in object hierarchy
            
        Returns:
            List of tuples (path, value1, value2) for differences
        """
        self.differences = []
        self._compare_recursive(obj1, obj2, path)
        return self.differences
    
    def _compare_recursive(self, obj1: Any, obj2: Any, path: str):
        """Internal recursive comparison."""
        
        # Check if path should be ignored
        if any(ignored in path for ignored in self.ignore_keys):
            return
        
        # Handle None cases
        if obj1 is None and obj2 is None:
            return
        if obj1 is None or obj2 is None:
            self.differences.append((path, obj1, obj2))
            return
        
        # Handle different types
        type1, type2 = type(obj1), type(obj2)
        if type1 != type2:
            # Special case: int vs float
            if (isinstance(obj1, (int, float)) and isinstance(obj2, (int, float))):
                if not np.isclose(float(obj1), float(obj2)):
                    self.differences.append((path, obj1, obj2))
                return
            self.differences.append((path, f"TYPE: {type1.__name__}", f"TYPE: {type2.__name__}"))
            return
        
        # Handle primitive types
        if isinstance(obj1, (bool, int, float, str)):
            if isinstance(obj1, float) and isinstance(obj2, float):
                if not np.isclose(obj1, obj2):
                    self.differences.append((path, obj1, obj2))
            elif obj1 != obj2:
                self.differences.append((path, obj1, obj2))
            return
        
        # Handle numpy arrays
        if isinstance(obj1, np.ndarray):
            if obj1.shape != obj2.shape:
                self.differences.append((path + ".shape", obj1.shape, obj2.shape))
            elif not np.allclose(obj1, obj2):
                self.differences.append((path, "array_differs", "array_differs"))
            return
        
        # Handle tuples
        if isinstance(obj1, tuple):
            if len(obj1) != len(obj2):
                self.differences.append((path + ".length", len(obj1), len(obj2)))
                return
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                self._compare_recursive(item1, item2, f"{path}[{i}]")
            return
        
        # Handle lists
        if isinstance(obj1, list):
            if len(obj1) != len(obj2):
                self.differences.append((path + ".length", len(obj1), len(obj2)))
                return
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                self._compare_recursive(item1, item2, f"{path}[{i}]")
            return
        
        # Handle dictionaries
        if isinstance(obj1, dict):
            keys1, keys2 = set(obj1.keys()), set(obj2.keys())
            
            # Keys only in obj1
            for key in keys1 - keys2:
                if not any(ignored in f"{path}.{key}" for ignored in self.ignore_keys):
                    self.differences.append((f"{path}.{key}", "EXISTS", "MISSING"))
            
            # Keys only in obj2
            for key in keys2 - keys1:
                if not any(ignored in f"{path}.{key}" for ignored in self.ignore_keys):
                    self.differences.append((f"{path}.{key}", "MISSING", "EXISTS"))
            
            # Compare common keys
            for key in keys1 & keys2:
                self._compare_recursive(obj1[key], obj2[key], f"{path}.{key}")
            return
        
        # Handle objects with __dict__
        if hasattr(obj1, "__dict__"):
            dict1, dict2 = obj1.__dict__, obj2.__dict__
            keys1, keys2 = set(dict1.keys()), set(dict2.keys())
            
            # Keys only in obj1
            for key in keys1 - keys2:
                if not any(ignored in f"{path}.{key}" for ignored in self.ignore_keys):
                    self.differences.append((f"{path}.{key}", "EXISTS", "MISSING"))
            
            # Keys only in obj2
            for key in keys2 - keys1:
                if not any(ignored in f"{path}.{key}" for ignored in self.ignore_keys):
                    self.differences.append((f"{path}.{key}", "MISSING", "EXISTS"))
            
            # Compare common keys
            for key in keys1 & keys2:
                self._compare_recursive(dict1[key], dict2[key], f"{path}.{key}")
            return
        
        # For other types, try direct comparison
        try:
            if obj1 != obj2:
                self.differences.append((path, obj1, obj2))
        except Exception:
            # If comparison fails, just note it
            self.differences.append((path, f"UNCOMPARABLE: {type1.__name__}", f"UNCOMPARABLE: {type2.__name__}"))


def format_value(val: Any, max_length: int = 100) -> str:
    """Format value for display."""
    if val is None:
        return "None"
    
    str_val = str(val)
    if len(str_val) > max_length:
        return str_val[:max_length] + "..."
    return str_val


def main():
    print("=" * 80)
    print("Configuration Comparison: Pickle vs Hydra")
    print("=" * 80)
    
    # Load from pickle
    print(f"\n[1/4] Loading configs from pickle: {args.pickle_path}")
    pickle_path = Path(args.pickle_path)
    env_cfg_pickle = load_pickle(str(pickle_path / "env.pkl"))
    agent_cfg_pickle = load_pickle(str(pickle_path / "agent.pkl"))
    print(f"  ✓ Loaded env.pkl")
    print(f"  ✓ Loaded agent.pkl")
    
    # Load from Hydra
    print(f"\n[2/4] Loading configs from Hydra:")
    print(f"  Task: {args.task_name}")
    print(f"  Entry point: {args.entry_point}")
    env_cfg_hydra, agent_cfg_hydra = register_task_to_hydra(args.task_name, args.entry_point)
    print(f"  ✓ Loaded env config via Hydra")
    print(f"  ✓ Loaded agent config via Hydra")
    
    # Compare environment configs
    print(f"\n[3/4] Comparing environment configs...")
    comparator = ConfigComparator(ignore_keys=args.ignore_keys)
    env_diffs = comparator.compare(env_cfg_pickle, env_cfg_hydra, "env_cfg")
    
    if env_diffs:
        print(f"  Found {len(env_diffs)} differences in environment config")
    else:
        print(f"  ✓ Environment configs are identical")
    
    # Compare agent configs
    print(f"\n[4/4] Comparing agent configs...")
    agent_diffs = comparator.compare(agent_cfg_pickle, agent_cfg_hydra, "agent_cfg")
    
    if agent_diffs:
        print(f"  Found {len(agent_diffs)} differences in agent config")
    else:
        print(f"  ✓ Agent configs are identical")
    
    # Generate report
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)
    
    output_lines = []
    
    if not env_diffs and not agent_diffs:
        msg = "✓ All configurations are identical!"
        print(f"\n{msg}")
        output_lines.append(msg)
    else:
        total_diffs = len(env_diffs) + len(agent_diffs)
        msg = f"Found {total_diffs} total differences ({len(env_diffs)} in env, {len(agent_diffs)} in agent)"
        print(f"\n{msg}")
        output_lines.append(msg)
        
        # Environment differences
        if env_diffs:
            header = "\n" + "=" * 80 + "\nENVIRONMENT CONFIG DIFFERENCES\n" + "=" * 80
            print(header)
            output_lines.append(header)
            
            for i, (path, val1, val2) in enumerate(env_diffs, 1):
                diff_str = f"\n{i}. Path: {path}"
                print(diff_str)
                output_lines.append(diff_str)
                
                pickle_str = f"   Pickle: {format_value(val1)}"
                print(pickle_str)
                output_lines.append(pickle_str)
                
                hydra_str = f"   Hydra:  {format_value(val2)}"
                print(hydra_str)
                output_lines.append(hydra_str)
        
        # Agent differences
        if agent_diffs:
            header = "\n" + "=" * 80 + "\nAGENT CONFIG DIFFERENCES\n" + "=" * 80
            print(header)
            output_lines.append(header)
            
            for i, (path, val1, val2) in enumerate(agent_diffs, 1):
                diff_str = f"\n{i}. Path: {path}"
                print(diff_str)
                output_lines.append(diff_str)
                
                pickle_str = f"   Pickle: {format_value(val1)}"
                print(pickle_str)
                output_lines.append(pickle_str)
                
                hydra_str = f"   Hydra:  {format_value(val2)}"
                print(hydra_str)
                output_lines.append(hydra_str)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\n✓ Report saved to: {args.output}")
    
    print("\n" + "=" * 80)
    
    # Close simulation
    simulation_app.close()
    
    # Return exit code based on whether there are differences
    sys.exit(0 if not env_diffs and not agent_diffs else 1)


if __name__ == "__main__":
    main()
