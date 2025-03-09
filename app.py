import streamlit as st
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.fftpack import fft, ifft
import scipy.signal as sps
import tempfile

###### SECTION I: COMPUTER VISION - MEDIAPIPE POSE CODE BASE ######
@st.cache_data
def replace_jump(signal, threshold):
    for n in range(1, len(signal)):
        if n >= 0 and n < len(signal) - 1:
            if np.abs(signal[n] - signal[n - 1]) > threshold * np.abs(signal[n + 1] - signal[n - 1]):
                previous_value = signal[n - 1]
                next_value = signal[n + 1]
                interpolated_value = (previous_value + next_value) / 2
                signal[n] = interpolated_value
    return signal

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = 180 - np.abs(radians * 180.0 / np.pi)
    return angle

@st.cache_data
def smooth_angles(angles, smoothing_factor=5):
    frame_numbers = np.array(range(len(angles)))
    spline = UnivariateSpline(frame_numbers, angles, k=5, s=smoothing_factor)
    return spline(frame_numbers)

def process_video(input_path, output_path, output_path2):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(smooth_landmarks=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.warning("Could not open the input video. Please check the file and try again.")
        return None, None, None, None, None, None, None, None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    out2 = cv2.VideoWriter(output_path2, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        st.warning(f"Could not create output video at {output_path}. Check file permissions or disk space.")
    if not out2.isOpened():
        st.warning(f"Could not create output video at {output_path2}. Check file permissions or disk space.")

    angles_data = []
    RSHDx, RSHDy = [], []
    RHIPx, RHIPy = [], []
    RKNEx, RKNEy = [], []
    RANKx, RANKy = [], []
    RHEEx, RHEEy = [], []
    RTOEx, RTOEy = [], []
    LTOEx, LTOEy = [], []
    frames_with_landmarks = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            frames_with_landmarks.append(frame)
            landmarks = results.pose_landmarks.landmark
            mp.solutions.drawing_utils.draw_landmarks(bgr_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            hip = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            )
            knee = -1 * calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            )
            ankle = -1 * calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            )

            text_hip = f"Right Hip: {hip:.1f} degrees"
            text_knee = f"Right Knee: {knee:.1f} degrees"
            text_ankle = f"Right Ankle: {ankle:.1f} degrees"
            cv2.rectangle(bgr_image, (30, 10), (340, 90), (0, 0, 0), -1)
            cv2.putText(bgr_image, text_hip, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bgr_image, text_knee, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bgr_image, text_ankle, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            RSHDx.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)
            RSHDy.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            RHIPx.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)
            RHIPy.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
            RKNEx.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)
            RKNEy.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
            RANKx.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
            RANKy.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
            RHEEx.append(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x)
            RHEEy.append(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y)
            RTOEx.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x)
            RTOEy.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y)
            LTOEy.append(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y)

            angles_data.append({
                'Time (s)': (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps,
                'Hip Flexion-Extension (deg)': hip,
                'Knee Flexion-Extension (deg)': knee,
                'Ankle Dorsiflexion-Plantarflexion (deg)': ankle
            })

        out.write(bgr_image)

    fRSHDx = replace_jump(RSHDx, 1.5)
    fRSHDy = replace_jump(RSHDy, 1.5)
    fRHIPx = replace_jump(RHIPx, 1.5)
    fRHIPy = replace_jump(RHIPy, 1.5)
    fRKNEx = replace_jump(RKNEx, 1.5)
    fRKNEy = replace_jump(RKNEy, 1.5)
    fRANKx = replace_jump(RANKx, 1.5)
    fRANKy = replace_jump(RANKy, 1.5)
    fRHEEx = replace_jump(RHEEx, 1.5)
    fRHEEy = replace_jump(RHEEy, 1.5)
    fRTOEx = replace_jump(RTOEx, 1.5)
    fRTOEy = replace_jump(RTOEy, 1.5)
    fLTOEy = replace_jump(LTOEy, 1.5)

    i = 0
    for i, frame2 in enumerate(frames_with_landmarks):
        hip = calculate_angle([fRSHDx[i], fRSHDy[i]], [fRHIPx[i], fRHIPy[i]], [fRKNEx[i], fRKNEy[i]])
        knee = calculate_angle([fRHIPx[i], fRHIPy[i]], [fRKNEx[i], fRKNEy[i]], [fRANKx[i], fRANKy[i]])
        ankle = calculate_angle([fRKNEx[i], fRKNEy[i]], [fRANKx[i], fRANKy[i]], [fRHEEx[i], fRHEEy[i]])

        text_hip = f"Right Hip: {hip:.1f} degrees"
        text_knee = f"Right Knee: {knee:.1f} degrees"
        text_ankle = f"Right Ankle: {ankle:.1f} degrees"
        cv2.rectangle(frame2, (30, 10), (340, 90), (0, 0, 0), -1)
        cv2.putText(frame2, text_hip, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame2, text_knee, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame2, text_ankle, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        rad = 5
        joints = [
            {"name": f"{hip:.1f}", "x": fRHIPx[i] * frame_width, "y": fRHIPy[i] * frame_height},
            {"name": f"{knee:.1f}", "x": fRKNEx[i] * frame_width, "y": fRKNEy[i] * frame_height},
            {"name": f"{ankle:.1f}", "x": fRANKx[i] * frame_width, "y": fRANKy[i] * frame_height},
        ]
        for joint in joints:
            cv2.circle(frame2, (int(joint["x"]), int(joint["y"])), rad, (0, 255, 0), -1)
            rect_x, rect_y = int(joint["x"] + 10), int(joint["y"] - 25)
            cv2.rectangle(frame2, (rect_x, rect_y), (rect_x + 50, rect_y + 20), (0, 0, 0), -1)
            cv2.putText(frame2, joint["name"], (int(joint["x"] + 10), int(joint["y"] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        connections = [(joints[0], joints[1]), (joints[1], joints[2])]
        for start, end in connections:
            cv2.line(frame2, (int(start["x"]), int(start["y"])), (int(end["x"]), int(end["y"])), (0, 255, 0), 2)

        circles = [
            {'radius': rad, 'center_x': fRSHDx[i] * frame_width, 'center_y': fRSHDy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRHIPx[i] * frame_width, 'center_y': fRHIPy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRKNEx[i] * frame_width, 'center_y': fRKNEy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRANKx[i] * frame_width, 'center_y': fRANKy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRHEEx[i] * frame_width, 'center_y': fRHEEy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRTOEx[i] * frame_width, 'center_y': fRTOEy[i] * frame_height, 'color': (0, 255, 0)},
        ]
        for circle in circles:
            cv2.circle(frame2, (int(circle['center_x']), int(circle['center_y'])), circle['radius'], circle['color'], -1)
        for j in range(0, len(circles) - 1):
            cv2.line(frame2, (int(circles[j]['center_x']), int(circles[j]['center_y'])), 
                     (int(circles[j + 1]['center_x']), int(circles[j + 1]['center_y'])), (0, 255, 0), 1)

        out2.write(frame2)
        i += 1

    st.write("To check that the videos have been processed successfully, ensure that there are some numbers for the sizes of the output videos below:")
    st.write(f"Output Video 1 size: {os.path.getsize(output_path)} bytes")
    st.write(f"Output Video 2 size: {os.path.getsize(output_path2)} bytes")

    angles_data = pd.DataFrame(angles_data)
    cap.release()
    out.release()
    out2.release()
    cv2.destroyAllWindows()

    if not angles_data.empty:
        return RTOEy, LTOEy, fRTOEy, fLTOEy, fps, angles_data, output_path, output_path2
    else:
        st.warning("No pose landmarks detected in the video. Download the CSV file to verify the data manually.")
        return None, None, None, None, None, None, None, None

###### SECTION II: DATA WRANGLING ######
@st.cache_data
def fourier_filter(data, sampling_freq, critical_freq):
    N = len(data)
    freqs = np.fft.fftfreq(N, d=1/sampling_freq)
    fft_values = fft(data)
    fft_values[np.abs(freqs) > critical_freq] = 0
    filtered_data = np.real(ifft(fft_values))
    return filtered_data

@st.cache_data
def extract_peaks(data):
    peak_indices, _ = sps.find_peaks(data)
    peaks = [(i, data[i]) for i in peak_indices]
    return peaks

@st.cache_data
def extract_troughs(data):
    trough_indices, _ = sps.find_peaks(-data)
    troughs = [(i, data[i]) for i in trough_indices]
    return troughs

###### SECTION III: STREAMLIT UI AND DATA PROCESSING ######
st.title("Gait Analysis Using Computer Vision")
st.write("This tool is powered by the ML solution MediaPipe Pose by Google.")
st.write("Try to keep the video within 15 seconds and use a resolution that is not too high as it may exceed available memory.")
st.write("The most accurate results shall be obtained from an ideal video containing a person **visible from head to toe** walking on a **treadmill**/with the **camera following the person** while walking [Click to View an Example from YouTube](https://www.youtube.com/watch?v=Mol0lrRBy3g)")

uploaded_file = st.file_uploader("Upload an MP4 File Containing a Person Walking.", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(uploaded_file.read())
        input_video_path = input_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as output_tmp1:
        output_video_path = output_tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as output_tmp2:
        output_video_path2 = output_tmp2.name

    RTOEy, LTOEy, fRTOEy, fLTOEy, fps, angles_data, final_output_path, final_output_path2 = process_video(
        input_video_path, output_video_path, output_video_path2
    )

    if RTOEy is not None:
        st.subheader("Processed Video with All Pose Landmarks")
        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            with open(final_output_path, "rb") as f:
                video_bytes = f.read()
            st.write("Click the button below to download the processed video (AVI format).")
            st.download_button(
                label="Download Video with All Pose Landmarks",
                data=video_bytes,
                file_name="processed_video_landmarks.avi",
                mime="video/x-msvideo"
            )
        else:
            st.warning("Processed video (landmarks) could not be generated correctly. Check the input video and try again.")

        st.subheader("Processed Video with Noise Corrected")
        if os.path.exists(final_output_path2) and os.path.getsize(final_output_path2) > 0:
            with open(final_output_path2, "rb") as f:
                video_bytes2 = f.read()
            st.write("Click the button below to download the processed video (AVI format).")
            st.download_button(
                label="Download Video with Noise Corrected",
                data=video_bytes2,
                file_name="processed_video_noise_corrected.avi",
                mime="video/x-msvideo"
            )
        else:
            st.warning("Processed video (noise corrected) could not be generated correctly. Check the input video and try again.")

        st.subheader("Right Toe Y-Coordinate Analysis")
        st.write('Further analysis by tracking movement of the right toe.')
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(np.arange(0, len(fRTOEy)) / fps, fRTOEy, label="Filtered Data", color="red", marker='x')
            ax.set_title("Right Toe Y-Coordinate (RAW DATA)")
            ax.set_xlabel("Time (seconds, Progression of Gait)")
            ax.set_ylabel("Y-Coordinate")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate the right toe y-coordinate plot due to an issue: {str(e)}. Download the CSV file below to plot it manually.")

        def find_dominant_frequency(data, sampling_freq):
            n = len(data)
            freq = np.fft.fftfreq(n, d=1/sampling_freq)
            fft_values = np.fft.fft(data)
            magnitude = np.abs(fft_values)
            positive_freqs = freq[:n // 2]
            positive_magnitude = magnitude[:n // 2]
            threshold_index = 1
            dominant_index = np.argmax(positive_magnitude[threshold_index:]) + threshold_index
            return positive_freqs[dominant_index]

        critical_freq = max(0.3, find_dominant_frequency(fRTOEy, fps))
        st.write(f"Suggested Critical (Walking) Frequency: {critical_freq:.2f} Hz")
        with st.expander("Click here to adjust if the cadence below does not seem right!"):
            st.write('A normal walking frequency is typically below 2.0 Hz')
            critical_freq = st.slider("Adjust Critical Frequency such that distinct peaks can be observed, use the new graph below to gauge.", 
                                      min_value=0.1, max_value=fps / 2, value=critical_freq, step=0.1)
            fRTOEy = fourier_filter(fRTOEy, fps, critical_freq)
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(np.arange(0, len(fRTOEy)) / fps, fRTOEy, label="Filtered Data", color="red", marker='x')
                ax.set_title("Right Toe Y-Coordinate (FILTERED DATA)")
                ax.set_xlabel("Time (seconds, Progression of Gait)")
                ax.set_ylabel("Y-Coordinate")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate the filtered right toe y-coordinate plot due to an issue: {str(e)}. Download the CSV file to plot it manually.")

        st.subheader("Select Range for Gait Analysis")
        st.write('Select range of the video where the **entire person can be seen from head to toe walking**.')
        time_values = np.arange(0, len(fRTOEy) / fps, 1 / fps)
        time_values = [f"{time:.2f} s (Frame {int(time * fps)})" for time in time_values]
        time_min = st.selectbox("Select minimum time (seconds):", options=time_values, index=0)
        time_max = st.selectbox("Select maximum time (seconds):", options=time_values, index=len(time_values) - 1)
        frame_min = int(float(time_min.split(" ")[0]) * fps)
        frame_max = int(float(time_max.split(" ")[0]) * fps)
        if frame_min > frame_max:
            st.warning("Minimum time must be less than or equal to maximum time. Adjust the selection and try again.")
        else:
            st.write(f"Selected range: {time_min} to {time_max}, {round(100 * abs(frame_max - frame_min) / (len(fRTOEy) - 1), 2)}% of the video is selected")

        st.subheader("Peak Detection")
        peaks = extract_peaks(fRTOEy)
        peak_df = pd.DataFrame(peaks, columns=["Frame Index", "Y-Coordinate"])
        filtered_peaks = peak_df[(peak_df["Frame Index"] >= frame_min) & (peak_df["Frame Index"] <= frame_max)]
        filtered_peaks = filtered_peaks.copy()
        filtered_peaks["Time (s)"] = filtered_peaks["Frame Index"] / fps
        columns_order = ["Time (s)"] + [col for col in filtered_peaks.columns if col != "Time (s)"]
        filtered_peaks = filtered_peaks[columns_order]
        if st.button('Click to display table of y-coordinate peaks for right toe'):
            st.write("Peaks within the selected range:")
            st.write(filtered_peaks)

        if len(filtered_peaks) >= 2:
            try:
                filtered_peaks['Step Length'] = filtered_peaks["Frame Index"].diff().fillna(0)
                stride_durations = [
                    filtered_peaks["Frame Index"].iloc[i + 1] - filtered_peaks["Frame Index"].iloc[i]
                    for i in range(len(filtered_peaks) - 1)
                ]
                st.subheader("Gait Characteristics")
                st.write("Here are some Gait Characteristics calculated using data within the time frame you have selected.")
                step_duration = round((filtered_peaks['Step Length'].mean()) / 2, 2)
                st.write(f"**Average Step Duration (Right Leg)**: {round(step_duration/fps, 3)} seconds")
                cycle_time = round(np.mean(stride_durations), 2)
                st.write(f"**Average Cycle Time (Right Leg)**: {round(cycle_time/fps, 3)} seconds")
                cadence = round(60 / (step_duration/fps), 1)
                st.write(f"**Cadence**: {cadence} steps / minute")
                num_of_steps = (len(fRTOEy)/fps) / ((filtered_peaks['Step Length'].mean()) / 2/fps)
                st.write(f"**Total Number of Steps**: {round(num_of_steps)} steps")
            except Exception as e:
                st.warning(f"Could not calculate gait characteristics due to an issue: {str(e)}. Download the CSV file to calculate them manually.")
        else:
            st.warning("Not enough peaks detected in the selected range to calculate gait characteristics. Adjust the time range or download the CSV file to analyze manually.")

        st.subheader('Joint Angles')
        st.write("Hover over the top right corner of the table:")
        st.write('Tip 1: Typing a negative sign(-) in the search box highlights all negative values.')
        st.write('Tip 2: The download button allows you to export the following data as a CSV file.')
        st.write(angles_data)

        try:
            angles_data['Hip Flexion-Extension (deg)'] = pd.to_numeric(angles_data['Hip Flexion-Extension (deg)'], errors='coerce')
            angles_data = angles_data.dropna(subset=['Hip Flexion-Extension (deg)'])
            critical_freq = max(0.3, find_dominant_frequency(angles_data['Hip Flexion-Extension (deg)'], fps))
            angles_data['Hip Flexion-Extension (deg)'] = fourier_filter(angles_data['Hip Flexion-Extension (deg)'].values, fps, critical_freq)
            sliced_data = angles_data['Hip Flexion-Extension (deg)'][frame_min:frame_max].reset_index(drop=True)
            peaks = extract_peaks(sliced_data)
            trough_df = pd.DataFrame(peaks, columns=["Frame Index", "Y-Coordinate"])

            def normalize_to_gait_cycle(data, troughs, num_points=100):
                cycles = []
                for i in range(len(troughs) - 1):
                    start, end = troughs[i][0], troughs[i + 1][0]
                    cycle_data = data[start:end]
                    normalized_cycle = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(cycle_data)), cycle_data)
                    cycles.append(normalized_cycle)
                return np.array(cycles)

            with st.expander("Click to Show Graph: Hip Flexion-Extension Across a Single Gait Cycle"):
                st.write("Note: Hip Extension: < 0°; Hip Flexion: > 0°")
                normalized_cycles = normalize_to_gait_cycle(angles_data['Hip Flexion-Extension (deg)'], peaks)
                mean_cycle = np.mean(normalized_cycles, axis=0)
                std_cycle = np.std(normalized_cycles, axis=0)
                x = np.linspace(0, 100, len(mean_cycle))
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x, mean_cycle, label="Mean Hip Angle", color="blue")
                ax.fill_between(x, mean_cycle - std_cycle, mean_cycle + std_cycle, color="blue", alpha=0.2, label="±1 STD")
                ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="0°")
                ax.set_xlabel("% Gait Cycle")
                ax.set_ylabel("Hip Flexion-Extension (degrees)")
                ax.set_title("Hip Angle Progression Across a Single Gait Cycle")
                ax.legend()
                st.pyplot(fig)
                st.write("Note: From Heel Strike (0%) to Toe Off (100%)")

            with st.expander("Click to Show Graph: Knee Flexion-Extension Across a Single Gait Cycle"):
                st.write("Note: Knee Extension: < 0°; Knee Flexion: > 0°")
                normalized_cycles = normalize_to_gait_cycle(angles_data['Knee Flexion-Extension (deg)'], peaks)
                mean_cycle = np.mean(normalized_cycles, axis=0)
                std_cycle = np.std(normalized_cycles, axis=0)
                x = np.linspace(0, 100, len(mean_cycle))
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x, mean_cycle, label="Mean Knee Angle", color="blue")
                ax.fill_between(x, mean_cycle - std_cycle, mean_cycle + std_cycle, color="blue", alpha=0.2, label="±1 STD")
                ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="0°")
                ax.set_xlabel("% Gait Cycle")
                ax.set_ylabel("Knee Flexion-Extension (degrees)")
                ax.set_title("Knee Angle Progression Across a Single Gait Cycle")
                ax.legend()
                st.pyplot(fig)
                st.write("Note: From Heel Strike (0%) to Toe Off (100%)")

            with st.expander("Click to Show Graph: Ankle Dorsiflexion-Plantarflexion Across a Single Gait Cycle"):
                st.write("Note: Ankle Plantarflexion: < 0°; Ankle Dorsiflexion: > 0°")
                normalized_cycles = normalize_to_gait_cycle(angles_data['Ankle Dorsiflexion-Plantarflexion (deg)'], peaks)
                mean_cycle = np.mean(normalized_cycles, axis=0)
                std_cycle = np.std(normalized_cycles, axis=0)
                x = np.linspace(0, 100, len(mean_cycle))
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x, mean_cycle, label="Mean Ankle Angle", color="blue")
                ax.fill_between(x, mean_cycle - std_cycle, mean_cycle + std_cycle, color="blue", alpha=0.2, label="±1 STD")
                ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="0°")
                ax.set_xlabel("% Gait Cycle")
                ax.set_ylabel("Ankle Dorsiflexion-Plantarflexion (degrees)")
                ax.set_title("Ankle Angle Progression Across a Single Gait Cycle")
                ax.legend()
                st.pyplot(fig)
                st.write("Note: From Heel Strike (0%) to Toe Off (100%)")
        except Exception as e:
            st.warning(f"Could not process joint angles or generate plots due to an issue: {str(e)}. Download the CSV file to analyze and plot manually.")

        st.subheader("Gait Symmetry using Robinson Index")
        st.write("Note: Inputs in this section will cause the application to load.")
        default_dist = float(num_of_steps * 0.7) if 'num_of_steps' in locals() else 0.0
        st.write(f"For a total of {round(num_of_steps) if 'num_of_steps' in locals() else 'unknown'} steps:")
        distance_input = st.number_input("Total distance traveled by person in meters (Default value calculated using right leg step length of 70cm):",
                                         min_value=0.0, value=default_dist, step=0.1)

        st.write("To calculate symmetry, we need to track the movement of the **left toe** too!")
        critical_freq = max(0.3, find_dominant_frequency(fLTOEy, fps))
        st.write(f"Suggested Critical (Walking) Frequency (Left Foot): {critical_freq:.2f} Hz")
        with st.expander("Click to adjust for the left toe like what was done for the right toe above"):
            st.write('A normal walking frequency is typically below 2.0 Hz')
            critical_freq = st.slider("Adjust Critical Frequency such that distinct peaks can be observed, use the new graph below to gauge",
                                      min_value=0.1, max_value=fps / 2, value=critical_freq, step=0.1)
            fLTOEy = fourier_filter(fLTOEy, fps, critical_freq)
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(np.arange(0, len(fLTOEy)) / fps, fLTOEy, label="Filtered Data", color="red", marker='x')
                ax.set_title("Left Toe Y-Coordinate (FILTERED DATA)")
                ax.set_xlabel("Time (seconds, Progression of Gait)")
                ax.set_ylabel("Y-Coordinate")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate the filtered left toe y-coordinate plot due to an issue: {str(e)}. Download the CSV file to plot it manually.")

        left_peaks = extract_peaks(fLTOEy)
        left_peak_df = pd.DataFrame(left_peaks, columns=["Frame Index", "Y-Coordinate"])
        left_filtered_peaks = left_peak_df[(left_peak_df["Frame Index"] >= frame_min) & (left_peak_df["Frame Index"] <= frame_max)]
        left_filtered_peaks = left_filtered_peaks.copy()
        left_filtered_peaks["Time (s)"] = left_filtered_peaks["Frame Index"] / fps
        columns_order = ["Time (s)"] + [col for col in left_filtered_peaks.columns if col != "Time (s)"]
        left_filtered_peaks = left_filtered_peaks[columns_order]
        if st.button('Click to display table of y-coordinate peaks for left toe'):
            st.write("Left Peaks within the selected range:")
            st.write(left_filtered_peaks)

        if len(left_filtered_peaks) >= 2:
            try:
                left_filtered_peaks['Step Length'] = left_filtered_peaks["Frame Index"].diff().fillna(0)
                left_stride_durations = [
                    left_filtered_peaks["Frame Index"].iloc[i + 1] - left_filtered_peaks["Frame Index"].iloc[i]
                    for i in range(len(left_filtered_peaks) - 1)
                ]
                st.write("Gait Characteristics for the left toe:")
                left_step_duration = round((left_filtered_peaks['Step Length'].mean()) / 2, 2)
                st.write(f"**Average Step Duration (Left Leg)**: {round(left_step_duration/fps, 3)} seconds")
                left_cycle_time = round(np.mean(left_stride_durations), 2)
                st.write(f"**Average Cycle Time (Left Leg)**: {round(left_cycle_time/fps, 3)} seconds")
            except Exception as e:
                st.warning(f"Could not calculate left leg gait characteristics due to an issue: {str(e)}. Download the CSV file to calculate them manually.")
        else:
            st.warning("Not enough peaks detected in the selected range for the left leg. Adjust the time range or download the CSV file to analyze manually.")

        st.write("Select the Affected Leg")
        affected_leg = st.radio("Choose which leg is affected:", options=["Left Leg", "Right Leg"])
        Xa_label = "Affected Left Leg Step Length" if affected_leg == "Left Leg" else "Right Leg Step Length"
        Xu_label = "Affected Right Leg Step Length" if affected_leg == "Left Leg" else "Left Leg Step Length"

        if 'cadence' in locals() and cadence > 0 and distance_input > 0:
            try:
                left_step_length = (left_step_duration/fps / (len(LTOEy)/fps)) * distance_input if 'left_step_duration' in locals() else 0
                right_step_length = distance_input / num_of_steps if 'num_of_steps' in locals() else 0
                st.write(f"**Left Leg Step Length:** {round(left_step_length, 5)} meters")
                st.write(f"**Right Leg Step Length:** {round(right_step_length, 5)} meters")

                Xa = left_step_length if affected_leg == "Left Leg" else right_step_length
                Xu = right_step_length if affected_leg == "Left Leg" else left_step_length
                SI = 100 * (Xa - Xu) / Xu

                st.write("""
                The **Robinson Index (SI)** is a measure of gait asymmetry, specifically focusing on the step length of the affected and unaffected legs. (Formula Reference: Robinson et al., 1987) 
                - \( Xa \): Step length of the **affected leg** (in meters).
                - \( Xu \): Step length of the **unaffected leg** (in meters).

                **Significance of SI**
                - **\( SI = 0 \):** Perfect symmetry; both legs have the same step length.
                - **\( SI > 0 \):** The affected leg has a longer step length than the unaffected leg (e.g. overcompensation).
                - **\( SI < 0 \):** The affected leg has a shorter step length than the unaffected leg (e.g. injury or reduced mobility).
                - **Magnitude of SI:** Larger values indicate greater asymmetry and more pronounced gait imbalance. Values close to 0 indicate normal variations.
                """)
                st.latex(r"Robinson~Index~SI = 100 \times \frac{X_a - X_u}{X_u}")
                st.write(f"## **Robinson Index (SI):** {SI:.2f}")
            except Exception as e:
                st.warning(f"Could not calculate Robinson Index due to an issue: {str(e)}. Download the CSV file to calculate it manually.")
        else:
            st.warning("Error. Adjust the time range or download the CSV file to analyze manually.")

        st.title("📊 EXPORT COMBINED DATA")
        st.markdown("""
        You can download a CSV containing the following information:
        - **Frame Index**
        - **Time**
        - **Y Coordinates** for movement of the left toe (Raw data)
        - **Y Coordinates** for movement of the left toe (Noise corrected)
        - **Y Coordinates** for movement of the right toe (Raw data)
        - **Y Coordinates** for movement of the right toe (Noise corrected)
        - **Right Hip Flexion-Extension Angle**
        - **Right Knee Flexion-Extension Angle**
        - **Right Ankle Dorsiflexion-Plantarflexion Angle**

        Click the button below to download the file for manual analysis if needed.
        """)
        combined_data = pd.DataFrame({
            "Frame Index": np.arange(len(RTOEy)),
            "Time (s)": np.arange(len(RTOEy)) / fps,
            "LTOEy (Raw)": LTOEy,
            "LTOEy (Noise Corrected)": fLTOEy,
            "RTOEy (Raw)": RTOEy,
            "RTOEy (Noise Corrected)": fRTOEy,
            "Right Hip Flexion-Extension Angle (deg)": angles_data["Hip Flexion-Extension (deg)"].values,
            "Right Knee Flexion-Extension Angle (deg)": angles_data["Knee Flexion-Extension (deg)"].values,
            "Right Ankle Dorsiflexion-Plantarflexion Angle (deg)": angles_data["Ankle Dorsiflexion-Plantarflexion (deg)"].values,
        })
        csv_data = combined_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Click to Export All Data to CSV",
            data=csv_data,
            file_name="gait_analysis_data.csv",
            mime="text/csv"
        )

st.info('Refresh the Page or click on \'Browse Files\' again to process the next video.')
st.warning('Note: Download your processed videos and CSV file before refreshing if you want to keep them.')
