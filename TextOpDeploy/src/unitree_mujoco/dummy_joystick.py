#!/usr/bin/env python3
"""
Dummy Joystick Simulator for /dev/input/js0
Creates a virtual joystick device using uinput to simulate joystick input.

Usage:
    sudo python dummy_joystick.py

Requirements:
    pip install python-uinput
    
    Or on Ubuntu/Debian:
    sudo apt-get install python3-uinput

System Requirements:
    - Load uinput kernel module: sudo modprobe uinput
    - Permissions: Run with sudo or add user to input group
"""

import time
import math
import argparse


def create_dummy_joystick_uinput():
    """Create a virtual joystick using python-uinput library."""
    try:
        import uinput
    except ImportError:
        print("ERROR: python-uinput not installed")
        print("Install with: pip install python-uinput")
        print("Or on Ubuntu/Debian: sudo apt-get install python3-uinput")
        return None

    # Define joystick capabilities
    # Standard joystick with 2 analog sticks (4 axes) and 12 buttons
    events = (
        uinput.BTN_JOYSTICK,
        uinput.ABS_X + (0, 255, 0, 0),      # Left stick X
        uinput.ABS_Y + (0, 255, 0, 0),      # Left stick Y
        uinput.ABS_RX + (0, 255, 0, 0),     # Right stick X
        uinput.ABS_RY + (0, 255, 0, 0),     # Right stick Y
        uinput.ABS_Z + (0, 255, 0, 0),      # Left trigger
        uinput.ABS_RZ + (0, 255, 0, 0),     # Right trigger
        uinput.BTN_TRIGGER,
        uinput.BTN_THUMB,
        uinput.BTN_THUMB2,
        uinput.BTN_TOP,
        uinput.BTN_TOP2,
        uinput.BTN_PINKIE,
        uinput.BTN_BASE,
        uinput.BTN_BASE2,
        uinput.BTN_BASE3,
        uinput.BTN_BASE4,
    )

    try:
        device = uinput.Device(events, name="Dummy-Joystick", bustype=uinput.BUS_USB)
        print(f"✓ Virtual joystick created: {device.name}")
        print(f"  Device should appear at /dev/input/js* (check with: ls -l /dev/input/js*)")
        return device
    except Exception as e:
        print(f"ERROR creating uinput device: {e}")
        print("\nTroubleshooting:")
        print("  1. Load uinput module: sudo modprobe uinput")
        print("  2. Check permissions: ls -l /dev/uinput")
        print("  3. Run with sudo or add user to input group:")
        print("     sudo usermod -a -G input $USER")
        return None


def create_dummy_joystick_evdev():
    """Create a virtual joystick using python-evdev library (alternative method)."""
    try:
        from evdev import UInput, ecodes as e, AbsInfo
    except ImportError:
        print("ERROR: python-evdev not installed")
        print("Install with: pip install evdev")
        return None

    # Define capabilities
    cap = {
        e.EV_KEY: [
            e.BTN_TRIGGER,
            e.BTN_THUMB,
            e.BTN_THUMB2,
            e.BTN_TOP,
            e.BTN_TOP2,
            e.BTN_PINKIE,
            e.BTN_BASE,
            e.BTN_BASE2,
            e.BTN_BASE3,
            e.BTN_BASE4,
        ],
        e.EV_ABS: [
            (e.ABS_X, AbsInfo(value=128, min=0, max=255, fuzz=0, flat=15, resolution=0)),
            (e.ABS_Y, AbsInfo(value=128, min=0, max=255, fuzz=0, flat=15, resolution=0)),
            (e.ABS_RX, AbsInfo(value=128, min=0, max=255, fuzz=0, flat=15, resolution=0)),
            (e.ABS_RY, AbsInfo(value=128, min=0, max=255, fuzz=0, flat=15, resolution=0)),
            (e.ABS_Z, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=15, resolution=0)),
            (e.ABS_RZ, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=15, resolution=0)),
        ],
    }

    try:
        device = UInput(cap, name="Dummy-Joystick-evdev", bustype=e.BUS_USB)
        print(f"✓ Virtual joystick created: {device.name}")
        print(f"  Device should appear at /dev/input/js* (check with: ls -l /dev/input/js*)")
        return device
    except Exception as e:
        print(f"ERROR creating evdev device: {e}")
        print("\nTroubleshooting:")
        print("  1. Load uinput module: sudo modprobe uinput")
        print("  2. Check permissions: ls -l /dev/uinput")
        print("  3. Run with sudo")
        return None


def simulate_joystick_input_uinput(device):
    """Simulate joystick input with circular motion pattern (uinput version)."""
    import uinput
    
    print("\n[Dummy Joystick] Simulating joystick input...")
    print("  Press Ctrl+C to stop")
    
    center = 128
    radius = 50
    angle = 0
    
    try:
        while True:
            # Circular motion on left stick
            x = int(center + radius * math.cos(angle))
            y = int(center + radius * math.sin(angle))
            
            # Send axis events
            device.emit(uinput.ABS_X, x)
            device.emit(uinput.ABS_Y, y)
            device.emit(uinput.ABS_RX, center)  # Right stick centered
            device.emit(uinput.ABS_RY, center)
            
            # Occasional button press
            if int(angle * 10) % 31 == 0:
                device.emit(uinput.BTN_TRIGGER, 1)
            else:
                device.emit(uinput.BTN_TRIGGER, 0)
            
            angle += 0.05
            if angle > 2 * math.pi:
                angle = 0
            
            time.sleep(0.02)  # 50 Hz update
            
    except KeyboardInterrupt:
        print("\n[Dummy Joystick] Stopped")


def simulate_joystick_input_evdev(device):
    """Simulate joystick input with circular motion pattern (evdev version)."""
    from evdev import ecodes as e
    
    print("\n[Dummy Joystick] Simulating joystick input...")
    print("  Press Ctrl+C to stop")
    
    center = 128
    radius = 50
    angle = 0
    
    try:
        while True:
            # Circular motion on left stick
            x = int(center + radius * math.cos(angle))
            y = int(center + radius * math.sin(angle))
            
            # Send axis events
            device.write(e.EV_ABS, e.ABS_X, x)
            device.write(e.EV_ABS, e.ABS_Y, y)
            device.write(e.EV_ABS, e.ABS_RX, center)
            device.write(e.EV_ABS, e.ABS_RY, center)
            
            # Occasional button press
            if int(angle * 10) % 31 == 0:
                device.write(e.EV_KEY, e.BTN_TRIGGER, 1)
            else:
                device.write(e.EV_KEY, e.BTN_TRIGGER, 0)
            
            device.syn()  # Sync events
            
            angle += 0.05
            if angle > 2 * math.pi:
                angle = 0
            
            time.sleep(0.02)  # 50 Hz update
            
    except KeyboardInterrupt:
        print("\n[Dummy Joystick] Stopped")


def keep_device_alive(device, library):
    """Keep the device alive without sending input (idle mode)."""
    print("\n[Dummy Joystick] Running in idle mode (no input simulation)")
    print("  Device will stay active at /dev/input/js*")
    print("  Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Dummy Joystick] Stopped")
    finally:
        if library == "evdev":
            device.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create a dummy joystick device at /dev/input/js*"
    )
    parser.add_argument(
        "--library",
        choices=["uinput", "evdev", "auto"],
        default="auto",
        help="Library to use for virtual device creation (default: auto)"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate joystick input with circular motion (default: idle mode)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check system requirements and exit"
    )
    
    args = parser.parse_args()
    
    # Check system requirements
    if args.check:
        print("=== System Requirements Check ===\n")
        
        import subprocess
        import os
        
        # Check if uinput module is loaded
        print("1. Checking uinput kernel module...")
        result = subprocess.run(["lsmod"], capture_output=True, text=True)
        if "uinput" in result.stdout:
            print("   ✓ uinput module is loaded")
        else:
            print("   ✗ uinput module NOT loaded")
            print("   → Load with: sudo modprobe uinput")
        
        # Check /dev/uinput permissions
        print("\n2. Checking /dev/uinput permissions...")
        if os.path.exists("/dev/uinput"):
            stat_info = os.stat("/dev/uinput")
            print(f"   ✓ /dev/uinput exists (mode: {oct(stat_info.st_mode)})")
            if os.access("/dev/uinput", os.W_OK):
                print("   ✓ Write access granted")
            else:
                print("   ✗ No write access")
                print("   → Run with sudo or add user to input group:")
                print("      sudo usermod -a -G input $USER")
        else:
            print("   ✗ /dev/uinput does not exist")
        
        # Check available libraries
        print("\n3. Checking Python libraries...")
        try:
            import uinput
            print("   ✓ python-uinput is installed")
        except ImportError:
            print("   ✗ python-uinput NOT installed")
            print("   → Install with: pip install python-uinput")
        
        try:
            import evdev
            print("   ✓ python-evdev is installed")
        except ImportError:
            print("   ✗ python-evdev NOT installed")
            print("   → Install with: pip install evdev")
        
        # Check existing joystick devices
        print("\n4. Existing joystick devices:")
        result = subprocess.run(["ls", "-l", "/dev/input/js*"], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("   No joystick devices found")
        
        return
    
    # Create virtual joystick
    print("=== Dummy Joystick Creator ===\n")
    
    device = None
    library_used = None
    
    if args.library == "auto":
        # Try uinput first, then evdev
        print("Trying python-uinput...")
        device = create_dummy_joystick_uinput()
        if device:
            library_used = "uinput"
        else:
            print("\nTrying python-evdev...")
            device = create_dummy_joystick_evdev()
            if device:
                library_used = "evdev"
    elif args.library == "uinput":
        device = create_dummy_joystick_uinput()
        library_used = "uinput"
    elif args.library == "evdev":
        device = create_dummy_joystick_evdev()
        library_used = "evdev"
    
    if device is None:
        print("\n✗ Failed to create virtual joystick")
        print("\nRun with --check to diagnose issues:")
        print("  python dummy_joystick.py --check")
        return 1
    
    # Run simulation or keep alive
    try:
        if args.simulate:
            if library_used == "uinput":
                simulate_joystick_input_uinput(device)
            else:
                simulate_joystick_input_evdev(device)
        else:
            keep_device_alive(device, library_used)
    finally:
        if library_used == "evdev":
            device.close()
        print("\n✓ Virtual joystick device closed")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
