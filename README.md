# ComputerVisionProjects
This Specific repository has all the ComputerVision projects

Hand tracking is a computer vision technique that detects and tracks the movement of hands using a camera. By leveraging frameworks like OpenCV, MediaPipe, we can implement hand tracking to control different functionalities such as:

Image Zooming Using Hand Gestures

The system detects both hands in the camera feed.
If the hands move apart, the image zooms in.
If the hands move closer, the image zooms out.
This works similarly to pinch-to-zoom on touchscreens but using hand gestures.
Volume Control Using Hand Distance

The system tracks the thumb and index finger of a single hand.
If the distance between them increases, the volume goes up.
If the distance decreases, the volume goes down.
The distance is mapped to a predefined volume range.
Implementation Approach
Use MediaPipe Hand Tracking to detect landmarks.
Measure the Euclidean distance between key points (like index finger and thumb).
Apply functions to control image scaling or adjust system volume.
Use OpenCV for real-time video processing.
