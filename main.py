
# import os
# import cv2
# import random
# import time
# import numpy as np
# from detector import PersonDetector
# from feature_extractor import FeatureExtractor
# from story_generator import StoryGenerator
# from tts import TextToSpeech

# from dotenv import load_dotenv
# load_dotenv()

# print("Print Model Name: ", os.getenv("DETECTOR_MODEL", 'yolov8n.pt'))

# def main():
#     print("Initializing components...")
    
#     # Initialize components
#     detector = PersonDetector(model_name=os.getenv("DETECTOR_MODEL"))
#     feature_extractor = FeatureExtractor()
#     story_gen = StoryGenerator(model="mistral")
#     tts = TextToSpeech()
    
#     # Test with either webcam or video file
#     video_source = 0  # Change to video path if needed
#     cap = cv2.VideoCapture(video_source)
    
#     if not cap.isOpened():
#         print("Error: Could not open video source")
#         return
    
#     # Configuration
#     capture_interval = 60  # seconds (reduced for testing)
#     last_capture_time = 0
#     tracking_active = False
    
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Process frame
#             current_time = time.time()
            
#             # 1. Detect and track people
#             try:
#                 annotated_frame, track_ids = detector.detect_and_track(frame)
#                 cv2.imshow("Live Tracking", annotated_frame)
#                 tracking_active = bool(track_ids)
#             except Exception as e:
#                 print(f"Detection error: {e}")
#                 annotated_frame = frame.copy()
#                 tracking_active = False
            
#             # 2. Periodically capture random person
#             if tracking_active and (current_time - last_capture_time >= capture_interval):
#                 last_capture_time = current_time
                
#                 try:
#                     # Select and extract random person
#                     person_img, bbox = detector.get_random_person(frame, track_ids)
#                     if person_img is None or person_img.size == 0:
#                         print("No valid person image extracted")
#                         continue
                    
#                     # Display extracted person
#                     cv2.imshow("Selected Person", person_img)
                    
#                     # 3. Extract features
#                     print ("Extracting features....")
#                     features = feature_extractor.extract(person_img)
                    
#                     # Add bounding box info (for story context)
#                     features["person_bbox"] = bbox
                    
#                     print("Features: ", dict(features))
                    
#                     # 4. Generate story
#                     story = story_gen.generate_story(features)
#                     print("\nGenerated Story:\n", story)
                    
#                     # 5. Text-to-speech
#                     tts.speak(story)
                    
#                     # Display story in console
#                     input("Press Enter to continue...")
                    
#                 except Exception as e:
#                     print(f"Processing error: {e}")
#                     continue
            
#             # Exit on 'q' key
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         print("Test completed")

# if __name__ == "__main__":
#     main()
# ////////////////////////////////////////////////////////////////////////////////



import os
import cv2
import random
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue

# Assuming these are in your project directory
from detector import PersonDetector
from feature_extractor import FeatureExtractor
from story_generator import StoryGenerator
from tts import TextToSpeech

from dotenv import load_dotenv
load_dotenv()

# Global variables for components (to be initialized once)
detector = None
feature_extractor = None
story_gen = None
tts = None

# Queue for inter-thread communication
person_data_queue = queue.Queue()

# Flags to control thread execution
running = True
story_window_active = False
story_completed_time = 0 # New global variable to track when story window closed

# Define the minimum delay between story presentations
MIN_STORY_DELAY = 5 # seconds

def initialize_components():
    """Initializes all the heavy components once."""
    global detector, feature_extractor, story_gen, tts
    print("Initializing components...")
    detector = PersonDetector(model_name=os.getenv("DETECTOR_MODEL", 'yolov8n.pt'))
    feature_extractor = FeatureExtractor()
    story_gen = StoryGenerator(model="mistral")
    tts = TextToSpeech()
    print("Components initialized.")

def update_frame(panel, frame):
    """Updates the Tkinter Label with the latest video frame."""
    try:
        # Convert the OpenCV image to a Tkinter PhotoImage
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk  # Keep a reference to prevent garbage collection
        panel.config(image=imgtk)
    except Exception as e:
        print(f"Error updating frame: {e}")

def video_processing_thread(video_source, live_tracking_panel, root):
    """
    Thread for video capture, detection, and putting data into a queue.
    """
    global running, story_window_active, story_completed_time
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source")
        messagebox.showerror("Video Error", "Could not open video source.")
        running = False
        return

    capture_interval = 60  # seconds (Original interval for person selection)
    last_person_selection_time = 0 # Track when a person was last selected

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # If a story window is active, pause processing and wait
        if story_window_active:
            time.sleep(0.1) # Small sleep to prevent busy-waiting
            continue

        current_time = time.time()

        # 1. Detect and track people
        annotated_frame = frame.copy()
        track_ids = []
        try:
            annotated_frame, track_ids = detector.detect_and_track(frame)
        except Exception as e:
            print(f"Detection error in thread: {e}")

        # Update the live tracking panel (on the main thread)
        root.after(1, update_frame, live_tracking_panel, annotated_frame)

        # 2. Periodically capture random person, but also respect the minimum story delay
        can_select_person = (current_time - last_person_selection_time >= capture_interval) and \
                            (current_time - story_completed_time >= MIN_STORY_DELAY)

        if track_ids and can_select_person:
            last_person_selection_time = current_time # Update this immediately to prevent rapid re-selection

            try:
                person_img, bbox = detector.get_random_person(frame, track_ids)
                if person_img is None or person_img.size == 0:
                    print("No valid person image extracted")
                    continue

                # 3. Extract features (can be done in this thread)
                print("Extracting features....")
                features = feature_extractor.extract(person_img)
                features["person_bbox"] = bbox

                # Put the person's data into the queue for the main thread
                person_data_queue.put((person_img, features))
                root.event_generate("<<PersonSelected>>", when="tail") # Notify main thread

            except Exception as e:
                print(f"Processing error in thread: {e}")

    cap.release()
    print("Video processing thread stopped.")

def show_person_details_window(parent_root, person_img, features):
    """
    Creates and displays a new Tkinter window for selected person's details.
    This function should be called on the main Tkinter thread.
    """
    global story_window_active
    story_window_active = True # Signal the video thread to pause

    details_window = tk.Toplevel(parent_root)
    details_window.title("Selected Person Details")
    details_window.geometry("800x600")
    # This protocol prevents the user from directly closing the window while story is being processed.
    # It will be re-enabled after the story is generated and presented.
    details_window.protocol("WM_DELETE_WINDOW", lambda: None) 

    # Display the selected person's image
    img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.thumbnail((300, 300)) # Resize for display
    imgtk = ImageTk.PhotoImage(image=img)
    person_image_label = tk.Label(details_window, image=imgtk)
    person_image_label.image = imgtk
    person_image_label.pack(pady=10)

    # Display features
    features_frame = ttk.LabelFrame(details_window, text="Extracted Features")
    features_frame.pack(padx=10, pady=5, fill="x")
    features_text = ""
    for key, value in features.items():
        if key == "person_bbox": # Special handling for bbox
            features_text += f"{key}: {value}\n"
        elif isinstance(value, np.ndarray):
            features_text += f"{key}: {value.shape} (Numpy Array)\n"
        else:
            features_text += f"{key}: {value}\n"
    features_label = ttk.Label(features_frame, text=features_text, wraplength=750, justify="left")
    features_label.pack(padx=5, pady=5)

    # Story display area
    story_frame = ttk.LabelFrame(details_window, text="Generated Story")
    story_frame.pack(padx=10, pady=5, fill="both", expand=True)
    story_text_widget = tk.Text(story_frame, wrap="word", height=10)
    story_text_widget.pack(padx=5, pady=5, fill="both", expand=True)
    story_text_widget.insert(tk.END, "Generating story...")
    story_text_widget.config(state="disabled") # Make it read-only initially

    # Add a "Speak Story" button
    # This button will be enabled after story generation is complete
    speak_button = ttk.Button(details_window, text="Speak Story", command=lambda: speak_story_async(story_text_widget.get("1.0", tk.END)), state="disabled")
    speak_button.pack(pady=10)

    # Function to close the details window and resume video
    def close_details_window(window):
        global story_window_active, story_completed_time
        story_window_active = False # Signal the video thread to resume
        story_completed_time = time.time() # Mark the time when the story window closed
        window.destroy()

    # Generate and speak story in a separate thread to avoid freezing GUI
    def generate_and_speak_story_and_close():
        try:
            story = story_gen.generate_story(features)
            print("\nGenerated Story:\n", story)
            
            # Update GUI with story (must be on main thread)
            parent_root.after(1, lambda: story_text_widget.config(state="normal"))
            parent_root.after(1, lambda: story_text_widget.delete("1.0", tk.END))
            parent_root.after(1, lambda: story_text_widget.insert(tk.END, story))
            parent_root.after(1, lambda: story_text_widget.config(state="disabled"))
            parent_root.after(1, lambda: speak_button.config(state="normal")) # Enable speak button
            
            # Re-enable the close protocol only after story is generated and displayed
            parent_root.after(1, lambda: details_window.protocol("WM_DELETE_WINDOW", lambda: close_details_window(details_window)))

            print("Starting speech...")
            tts.speak(story) # This call will now block until audio finishes
            print("Speech finished.")
            
        except Exception as e:
            print(f"Error generating or speaking story: {e}")
            parent_root.after(1, lambda: story_text_widget.config(state="normal"))
            parent_root.after(1, lambda: story_text_widget.delete("1.0", tk.END))
            parent_root.after(1, lambda: story_text_widget.insert(tk.END, f"Error: {e}"))
            parent_root.after(1, lambda: story_text_widget.config(state="disabled"))
            parent_root.after(1, lambda: speak_button.config(state="disabled")) # Disable speak button on error
        finally:
            # Ensure the window closes after speech completes, even if there was an error in TTS.
            # This is called via after() to ensure it runs on the main thread after the blocking tts.speak()
            parent_root.after(1, lambda: close_details_window(details_window))


    # Start story generation and speaking in a new thread
    threading.Thread(target=generate_and_speak_story_and_close, daemon=True).start()

def speak_story_async(story_text):
    """Speaks the story in a separate thread so GUI remains responsive if button is pressed again."""
    threading.Thread(target=lambda: tts.speak(story_text), daemon=True).start()


def handle_person_selection(root, live_tracking_panel):
    """
    Callback function when a person is selected in the video thread.
    This runs on the main Tkinter thread.
    """
    if not person_data_queue.empty():
        person_img, features = person_data_queue.get()
        show_person_details_window(root, person_img, features)

def on_closing(root, video_thread):
    """Handles proper shutdown when the main window is closed."""
    global running
    print("Closing application...")
    running = False # Signal the video thread to stop
    if video_thread.is_alive(): # Check if the thread is still running before joining
        video_thread.join() # Wait for the video thread to finish
    cv2.destroyAllWindows()
    root.destroy()
    print("Application closed.")

def main():
    initialize_components()

    root = tk.Tk()
    root.title("Live Person Tracking")

    # Window 1: Live Tracking
    live_tracking_panel = tk.Label(root)
    live_tracking_panel.pack(padx=10, pady=10)

    # Start the video processing in a separate thread
    video_source = 0  # Change to video path if needed
    video_thread = threading.Thread(target=video_processing_thread,
                                    args=(video_source, live_tracking_panel, root),
                                    daemon=True) # Daemon so it exits with main thread
    video_thread.start()

    # Bind the custom event for person selection
    root.bind("<<PersonSelected>>", lambda event: handle_person_selection(root, live_tracking_panel))

    # Handle window closing gracefully
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, video_thread))

    root.mainloop()

if __name__ == "__main__":
    main()