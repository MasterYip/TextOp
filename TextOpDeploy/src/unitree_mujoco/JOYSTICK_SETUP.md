# Dummy Joystick Setup Guide

This guide explains how to create a virtual joystick device at `/dev/input/js0` for testing purposes.

## Quick Start

```bash
# Install evdev (recommended)
sudo apt-get install python3-evdev

# OR Install uinput (not works)
sudo apt-get install python3-uinput
sudo modprobe uinput

# Run the dummy joystick
sudo python dummy_joystick.py
```

## Detailed Setup

### Prerequisites

1. **Load the uinput kernel module:**
   ```bash
   sudo modprobe uinput
   ```
   
   To load automatically on boot:
   ```bash
   echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf
   ```

2. **Install Python library (choose one):**

   Option A - evdev (recommended):
   ```bash
   pip install evdev
   ```
   
   Option B - python-uinput:
   ```bash
   pip install python-uinput
   # OR on Ubuntu/Debian:
   sudo apt-get install python3-uinput
   ```

3. **Set permissions (optional, to avoid sudo):**
   ```bash
   # Add user to input group
   sudo usermod -a -G input $USER
   
   # Create udev rule for /dev/uinput
   echo 'KERNEL=="uinput", MODE="0660", GROUP="input"' | \
       sudo tee /etc/udev/rules.d/99-uinput.rules
   
   # Reload udev rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   
   # Log out and back in for group changes to take effect
   ```

## Usage

### Check System Requirements
```bash
python dummy_joystick.py --check
```

### Create Idle Joystick (No Input)
Creates a virtual joystick that stays active but doesn't send input:
```bash
sudo python dummy_joystick.py
```

### Create Joystick with Simulated Input
Creates a virtual joystick and simulates circular motion:
```bash
sudo python dummy_joystick.py --simulate
```

### Specify Library
```bash
# Use evdev library
sudo python dummy_joystick.py --library evdev

# Use uinput library
sudo python dummy_joystick.py --library uinput

# Auto-detect (default)
sudo python dummy_joystick.py --library auto
```

## Verification

### Check if joystick device was created:
```bash
ls -l /dev/input/js*
```

### Test with jstest (if available):
```bash
sudo apt-get install joystick
jstest /dev/input/js0
```

### Monitor joystick events:
```bash
# Using evtest
sudo apt-get install evtest
sudo evtest /dev/input/js0

# Using cat (raw data)
sudo cat /dev/input/js0 | od -x
```

## Troubleshooting

### "Permission denied" error
- Run with `sudo`
- OR add user to input group (see Prerequisites step 3)

### "No such file or directory: /dev/uinput"
- Load uinput module: `sudo modprobe uinput`

### "Module uinput not found"
- Install uinput support: `sudo apt-get install linux-modules-extra-$(uname -r)`
- OR rebuild kernel with CONFIG_INPUT_UINPUT=m

### Device not appearing at /dev/input/js0
- Check if other joystick devices exist: `ls /dev/input/js*`
- Your device might be js1, js2, etc.
- Unplug physical joysticks or stop other virtual joystick processes

### "ImportError: No module named 'evdev'" or "'uinput'"
- Install the required library (see Prerequisites step 2)

## Integration with Your Application

Once the dummy joystick is running, it will appear as a standard joystick device at `/dev/input/js0` (or js1, js2, etc.). Your application can read from it like any normal joystick.

Keep the `dummy_joystick.py` script running in the background while your application uses the virtual joystick:

```bash
# Terminal 1: Start dummy joystick
sudo python dummy_joystick.py

# Terminal 2: Run your application
python your_application.py
```

## Stopping the Dummy Joystick

Press `Ctrl+C` in the terminal where `dummy_joystick.py` is running. The virtual device will be automatically cleaned up.
