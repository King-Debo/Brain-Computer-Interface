# Import the necessary libraries and frameworks
import mne
import nilearn
import pynirs
import pyopto

# Initialize the variables and parameters
data = None # The brain signals and behavioral responses from the participants
device = None # The EEG, fMRI, NIRS, or optogenetics device
sensor = None # The sensor or electrode that is attached to the participant's head
channel = None # The channel or frequency that is used to record the brain signals
sample_rate = None # The sample rate or resolution that is used to record the brain signals
duration = None # The duration or length of the recording session

# Define the functions for the data collection
def select_device():
    # This function allows the user to select the desired device from a list of available devices
    global device
    print("Please select the device that you want to use to record the brain signals and behavioral responses from the participants.")
    print("The available devices are: EEG, fMRI, NIRS, or optogenetics.")
    device = input("Enter the name of the device: ")
    print(f"You have selected {device} as your device.")

def select_sensor():
    # This function allows the user to select the desired sensor or electrode from a list of available sensors or electrodes on the device
    global device, sensor
    print(f"Please select the sensor or electrode that you want to use on your {device}.")
    if device == "EEG":
        print("The available sensors are: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2, or A1.")
    elif device == "fMRI":
        print("The available sensors are: S1, S2, S3, S4, S5, S6, S7, S8, S9, or S10.")
    elif device == "NIRS":
        print("The available sensors are: N1, N2, N3, N4, N5, N6, N7, N8, N9, or N10.")
    elif device == "optogenetics":
        print("The available sensors are: O1, O2, O3, O4, O5, O6, O7, O8, O9, or O10.")
    else:
        print("Invalid device. Please select a valid device.")
        return
    sensor = input("Enter the name of the sensor or electrode: ")
    print(f"You have selected {sensor} as your sensor or electrode.")

def select_channel():
    # This function allows the user to select the desired channel or frequency from a list of available channels or frequencies on the device
    global device, channel
    print(f"Please select the channel or frequency that you want to use on your {device}.")
    if device == "EEG":
        print("The available channels are: alpha, beta, gamma, delta, or theta.")
    elif device == "fMRI":
        print("The available channels are: BOLD, ASL, or DTI.")
    elif device == "NIRS":
        print("The available channels are: 760 nm, 850 nm, or 940 nm.")
    elif device == "optogenetics":
        print("The available channels are: 470 nm, 530 nm, or 590 nm.")
    else:
        print("Invalid device. Please select a valid device.")
        return
    channel = input("Enter the name of the channel or frequency: ")
    print(f"You have selected {channel} as your channel or frequency.")

def select_sample_rate():
    # This function allows the user to select the desired sample rate or resolution from a list of available sample rates or resolutions on the device
    global device, sample_rate
    print(f"Please select the sample rate or resolution that you want to use on your {device}.")
    if device == "EEG":
        print("The available sample rates are: 100 Hz, 200 Hz, 500 Hz, or 1000 Hz.")
    elif device == "fMRI":
        print("The available sample rates are: 1 s, 2 s, 5 s, or 10 s.")
    elif device == "NIRS":
        print("The available sample rates are: 10 Hz, 20 Hz, 50 Hz, or 100 Hz.")
    elif device == "optogenetics":
        print("The available sample rates are: 10 Hz, 20 Hz, 50 Hz, or 100 Hz.")
    else:
        print("Invalid device. Please select a valid device.")
        return
    sample_rate = input("Enter the name of the sample rate or resolution: ")
    print(f"You have selected {sample_rate} as your sample rate or resolution.")

def select_duration():
    # This function allows the user to select the desired duration or length of the recording session
    global duration
    print("Please select the duration or length of the recording session.")
    print("The available durations are: 10 s, 20 s, 30 s, or 60 s.")
    duration = input("Enter the name of the duration or length: ")
    print(f"You have selected {duration} as your duration or length.")

def connect_device():
    # This function connects and communicates with the device, sensor, and electrode, and checks the status and functionality
    global device, sensor, channel, sample_rate
    print("Connecting and communicating with the device, sensor, and electrode...")
    # TODO: Replace the device, sensor, channel, and sample_rate names and values with your own
    if device == "EEG":
        device = mne.io.read_raw_eeg("device/eeg_device.fif")
        sensor = device.pick_channels(["Fp1"])
        channel = device.filter(l_freq=8, h_freq=12)
        sample_rate = device.resample(sfreq=100)
    elif device == "fMRI":
        device = nilearn.image.load_img("device/fmri_device.nii.gz")
        sensor = device.slicer[0, 0, 0]
        channel = device.math_img("np.mean(img, axis=-1)", img=device)
        sample_rate = device.resample_to_img(target_img=device, interpolation="nearest")
    elif device == "NIRS":
        device = pynirs.io.load_nirs("device/nirs_device.snirf")
        sensor = device.probe.link.source[0]
        channel = device.probe.wavelengths[0]
        sample_rate = device.data.sampling_rate
    elif device == "optogenetics":
        device = pyopto.device("device/opto_device")
        sensor = device.get_sensor(0)
        channel = device.get_channel(0)
        sample_rate = device.get_sample_rate()
    else:
        print("Invalid device. Please select a valid device.")
        return
    print("The device, sensor, and electrode are connected and communicated.")

def collect_data():
    # This function collects and preprocesses the brain signals and behavioral responses from the participants, using the device, sensor, and electrode
    global device, sensor, channel, sample_rate, duration, data
    print("Collecting and preprocessing the brain signals and behavioral responses from the participants...")
    # TODO: Replace the data collection and preprocessing steps with your own
    data = device.record(duration=duration, save=True, filename="data/data.fif")
    data = data.filter(l_freq=1, h_freq=40)
    data = data.notch_filter(freqs=[50, 100])
    data = data.crop(tmin=0, tmax=duration)
    data = data.apply_ica()
    data = data.get_data()
    data = data.reshape(-1, 64)
    data = data / data.max()
    print("The brain signals and behavioral responses from the participants are collected and preprocessed.")
