import socket
import threading
import time
import queue
from hexapodkinematics import inverse_kinematics, arminverse_kinematics
from tkinter import ttk
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import io
from tkinter import messagebox
import os
import onnxruntime as ort
import multiprocessing as mp
from ultralytics import YOLO


root = tk.Tk()
HOST = '192.168.2.200'
PORT = 5000


class VideoStream:
    def __init__(self, root, robot_ip, port=8000):
        self.robot_ip = robot_ip
        self.port = port
        self.is_streaming = False
        self.stream_thread = None
        self.detection_thread = None
        self.object_detection_enabled = False
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        
        # Use larger queue for raw frames but smaller display queue
        self.frame_queue = queue.Queue(maxsize=10)  # Increased buffer
        self.display_queue = queue.Queue(maxsize=1)  # Only need the latest frame
        
        # Frame skip counter for object detection to reduce CPU load
        self.frame_skip = 0
        self.skip_count = 1  # Only process every 3rd frame for detection
        
        # Create a frame to hold the video
        self.frame = tk.Frame(root)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create a canvas for displaying the video
        self.canvas = tk.Canvas(self.frame, bg="black", width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create control buttons
        self.control_frame = tk.Frame(self.frame)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.btn_start = tk.Button(self.control_frame, text="Start Camera", command=self.start_stream)
        self.btn_start.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_stop = tk.Button(self.control_frame, text="Stop Camera", command=self.stop_stream, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add object detection toggle button
        self.detect_var = tk.BooleanVar(value=False)
        self.btn_detect = tk.Checkbutton(self.control_frame, text="Enable Object Detection", 
                                        variable=self.detect_var, command=self.toggle_detection)
        self.btn_detect.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Quality control slider
        self.quality_label = tk.Label(self.control_frame, text="Quality:")
        self.quality_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.quality_var = tk.IntVar(value=1)  # 0=low, 1=medium, 2=high
        self.quality_scale = tk.Scale(self.control_frame, from_=0, to=2, orient=tk.HORIZONTAL,
                                    variable=self.quality_var, showvalue=False, length=100,
                                    command=self.update_quality)
        self.quality_scale.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status and FPS labels
        self.status_var = tk.StringVar(value="Camera Off")
        self.status_label = tk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        self.fps_label = tk.Label(self.control_frame, textvariable=self.fps_var)
        self.fps_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Pre-allocate resources
        self.bytes_io = io.BytesIO()
        self.current_image = None
        self.photo_image = None
        
        # Caching for performance optimization
        self.last_canvas_width = 640
        self.last_canvas_height = 480
        self.last_resize_dimensions = (640, 480)
        
        # Threading optimizations
        self.running = True  # Flag for clean thread shutdown
        
        # Reduced quality for faster processing but still decent results
        self.detection_size = (224, 224)  # Smaller model input size
        self.resize_quality = Image.NEAREST  # Faster resize algorithm
        
        # Initialize object detection model
        self.initialize_object_detection()
        
        # Start UI update loop
        self.update_ui()
    
    def send_command(self, command):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(1.0)  # Reduced timeout for faster error recovery
            try:
                s.sendto(command.encode(), (self.robot_ip, 5000))
                response, _ = s.recvfrom(1024)
                return response.decode()
            except Exception as e:
                print(f"Error sending camera command: {e}")
                return None
    
    def start_stream(self):
        response = self.send_command("START_CAMERA")
        if response == "CAMERA_STARTED" or True:  # Allow continuing even if response is unexpected
            self.is_streaming = True
            self.running = True
            
            # Use daemon threads for automatic cleanup
            self.stream_thread = threading.Thread(target=self.receive_stream)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            self.detection_thread = threading.Thread(target=self.detection_worker)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.status_var.set("Camera On")
        else:
            self.status_var.set("Failed to start camera")
    
    def stop_stream(self):
        self.send_command("STOP_CAMERA")
        self.is_streaming = False
        self.running = False  # Signal threads to terminate
        
        # Empty queues
        self.clear_queues()
        
        # No need to join threads with daemon=True
        self.stream_thread = None
        self.detection_thread = None
        
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("Camera Off")
        self.fps_var.set("FPS: 0")
        
        # Clear canvas
        self.canvas.delete("all")
    
    def clear_queues(self):
        """Empty all queues without blocking"""
        try:
            while True:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
        except queue.Empty:
            pass
            
        try:
            while True:
                self.display_queue.get_nowait()
                self.display_queue.task_done()
        except queue.Empty:
            pass
    
    def update_quality(self, value):
        """Update the quality settings"""
        quality = int(value)
        if quality == 0:  # Low
            self.detection_size = (160, 160)
            self.resize_quality = Image.NEAREST
            self.skip_count = 1  # Process every 4th frame
        elif quality == 1:  # Medium
            self.detection_size = (224, 224)
            self.resize_quality = Image.NEAREST
            self.skip_count = 1  # Process every 3rd frame
        else:  # High
            self.detection_size = (320, 320)
            self.resize_quality = Image.BILINEAR  # Better quality but still fast
            self.skip_count = 1  # Process every 2nd frame
    
    def receive_stream(self):
        """Thread function for receiving video frames - optimized version"""
        video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            video_socket.connect((self.robot_ip, self.port))
            video_socket.settimeout(0.5)  # Shorter timeout
            # Set TCP_NODELAY to disable Nagle's algorithm for lower latency
            video_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            print(f"Failed to connect to video stream: {e}")
            self.frame.after(0, lambda: self.status_var.set("Connection Failed"))
            self.frame.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
            self.frame.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))
            self.is_streaming = False
            return
        
        # Use pre-allocated buffer with larger size for better throughput
        buffer = bytearray(65536)
        buffer_view = memoryview(buffer)  # Avoid unnecessary copying
        buffer_size = 0
        
        # Receive and process video frames
        while self.running and self.is_streaming:
            try:
                # Receive data directly into our buffer to avoid copies
                bytes_read = video_socket.recv_into(buffer_view[buffer_size:], 65536 - buffer_size)
                if not bytes_read:
                    # Connection closed
                    time.sleep(0.01)
                    continue
                
                buffer_size += bytes_read
                
                # Process all complete frames in buffer
                processed_bytes = 0
                while buffer_size >= 4:
                    # Extract frame size
                    frame_size = int.from_bytes(buffer[:4], byteorder='big')
                    
                    # Check if we have a complete frame
                    if buffer_size >= frame_size + 4:
                        # If the queue is full, drop the oldest frame to make room
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.task_done()
                            except queue.Empty:
                                pass
                        
                        # Extract frame data and add to queue
                        frame_data = bytes(buffer[4:frame_size+4])
                        try:
                            self.frame_queue.put(frame_data, block=False)
                        except queue.Full:
                            pass  # Skip frame if queue is still full
                        
                        # Move remaining data to start of buffer
                        buffer_size -= (frame_size + 4)
                        if buffer_size > 0:
                            buffer_view[:buffer_size] = buffer_view[frame_size+4:frame_size+4+buffer_size]
                    else:
                        # Need more data
                        break
                        
            except socket.timeout:
                # Just a timeout, continue
                continue
            except Exception as e:
                if self.is_streaming:
                    print(f"Error receiving video: {e}")
                break
        
        # Clean up
        try:
            video_socket.close()
        except:
            pass
    
    def detection_worker(self):
        """Thread function for processing frames - optimized version"""
        while self.running and self.is_streaming:
            try:
                # Get frame from queue with timeout
                frame_bytes = self.frame_queue.get(timeout=0.1)
                
                try:
                    # Process frame
                    self.bytes_io.seek(0)
                    self.bytes_io.truncate(0)
                    self.bytes_io.write(frame_bytes)
                    self.bytes_io.seek(0)
                    
                    # Decode image
                    image = Image.open(self.bytes_io)
                    image.load()
                    image = image.rotate(180)
                    
                    # Apply object detection only on some frames to improve performance
                    if self.object_detection_enabled:
                        self.frame_skip += 1
                        if self.frame_skip > self.skip_count:
                            image = self.detect_objects(image)
                            self.frame_skip = 0
                    
                    # Skip adding to display queue if it's already full
                    if not self.display_queue.full():
                        self.display_queue.put(image, block=False)
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                
                finally:
                    self.frame_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_streaming:
                    print(f"Error in detection worker: {e}")
                time.sleep(0.01)
    
    def update_ui(self):
        """Update the UI with new frames - optimized version"""
        try:
            # Update FPS counter every second
            current_time = time.time()
            elapsed = current_time - self.last_time
            
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.fps_var.set(f"FPS: {self.fps:.1f}")
                self.last_time = current_time
                self.frame_count = 0
            
            # Update canvas with new image if available
            if not self.display_queue.empty():
                try:
                    new_image = self.display_queue.get_nowait()
                    self.update_canvas(new_image)
                    self.frame_count += 1
                    self.display_queue.task_done()
                except queue.Empty:
                    pass
                
        except Exception as e:
            print(f"Error updating UI: {e}")
            
        # Schedule next update - use variable rate based on FPS
        # Use faster updates when FPS is higher
        update_rate = int(1000 / min(max(self.fps, 15), 60)) if self.fps > 0 else 20
        self.frame.after(update_rate, self.update_ui)
    
    def update_canvas(self, image):
        """Update the canvas with a new image - optimized version"""
        try:
            # Get canvas dimensions - cache these to avoid frequent lookups
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Only calculate resize when dimensions change significantly
            resize_needed = False
            if (abs(canvas_width - self.last_canvas_width) > 10 or 
                abs(canvas_height - self.last_canvas_height) > 10 or
                canvas_width <= 1 or canvas_height <= 1):
                
                self.last_canvas_width = max(1, canvas_width)
                self.last_canvas_height = max(1, canvas_height)
                
                # Calculate aspect-preserving dimensions
                img_width, img_height = image.size
                aspect_ratio = img_width / img_height
                
                if canvas_width / canvas_height > aspect_ratio:
                    new_height = canvas_height
                    new_width = int(aspect_ratio * new_height)
                else:
                    new_width = canvas_width
                    new_height = int(new_width / aspect_ratio)
                
                self.last_resize_dimensions = (new_width, new_height)
                resize_needed = True
            
            # Only resize when needed
            if resize_needed:
                image = image.resize(self.last_resize_dimensions, self.resize_quality)
            
            # Delete old reference first
            del self.photo_image
            
            # Update image
            self.current_image = image
            self.photo_image = ImageTk.PhotoImage(image=self.current_image)
            
            # Clear canvas and create new image
            self.canvas.delete("all")
            self.canvas.create_image(
                self.last_canvas_width // 2, self.last_canvas_height // 2,
                image=self.photo_image, anchor=tk.CENTER
            )
            
        except Exception as e:
            print(f"Error updating canvas: {e}")

    def initialize_object_detection(self):
        """Initialize object detection with optimizations"""
        try:
            # Load COCO class names
            with open('coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Create a list of indices for person and bottle classes
            self.target_classes = ['bottle']
            self.target_class_indices = [i for i, cls in enumerate(self.classes) if cls in self.target_classes]
            
            # Load pre-trained YOLOv4-tiny model with optimizations
            self.net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
            
            # Try GPU acceleration
            backend_set = False
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # Use FP16 for faster inference
                self.status_var.set("Using GPU acceleration (FP16)")
                backend_set = True
            except:
                pass
                
            if not backend_set:
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.status_var.set("Using CPU optimization")
                except:
                    self.status_var.set("Object detection initialized")
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            if cv2.__version__.startswith('4'):
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            else:
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Use distinct colors for our target classes
            self.colors = {
                'person': (0, 255, 0),  # Green for people
                'bottle': (0, 0, 255)   # Blue for bottles
            }
            # Prepare optimized confidence and NMS thresholds
            self.conf_threshold = 0.3
            self.nms_threshold = 0.4
            
        except Exception as e:
            print(f"Error initializing object detection: {e}")
            self.status_var.set("Object detection failed to initialize")
    
    def toggle_detection(self):
        """Toggle object detection with status update"""
        self.object_detection_enabled = self.detect_var.get()
        if self.object_detection_enabled:
            quality = "Low" if self.quality_var.get() == 0 else "Medium" if self.quality_var.get() == 1 else "High"
            self.status_var.set(f"Detecting people & bottles ({quality})")
        else:
            self.status_var.set("Detection OFF")

    def detect_objects(self, image):
        """Optimized object detection for people and bottles only"""
        # Convert PIL image to OpenCV format
        opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = opencvImage.shape[:2]
        
        # Create a blob with optimized size for speed
        blob = cv2.dnn.blobFromImage(opencvImage, 1/255.0, self.detection_size, 
                                    swapRB=True, crop=False)
        
        # Forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections with optimized filtering
        class_ids = []
        confidences = []
        boxes = []
        
        # Process only the most confident detections and filter for person/bottle
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only process if it's a person or bottle
                class_name = self.classes[class_id]
                if class_name in self.target_classes and confidence > self.conf_threshold:
                    # Convert from center coordinates to top-left
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression to remove redundant boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        # Return original image if no detections
        if len(boxes) == 0 or len(indices) == 0:
            return image
        
        # Draw bounding boxes
        indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        
        for i in indices:
            x, y, w, h = boxes[i]
            class_name = self.classes[class_ids[i]]
            confidence = confidences[i]
            
            # Get color for this class
            color = self.colors[class_name]
            
            # Ensure coordinates are within image boundaries
            x = max(0, x)
            y = max(0, y)
            x_max = min(width, x + w)
            y_max = min(height, y + h)
            
            # Draw rectangle with thickness based on image size
            thickness = max(1, int(min(width, height) / 400))
            cv2.rectangle(opencvImage, (x, y), (x_max, y_max), color, thickness)
            
            # Draw label
            text = f"{class_name} {confidence:.2f}"
            font_scale = max(0.4, min(width, height) / 1000)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            
            # Create background for text
            cv2.rectangle(opencvImage, (x, y - text_size[1] - 10), 
                        (x + text_size[0] + 10, y), color, -1)
            
            # Draw text (white)
            cv2.putText(opencvImage, text, (x + 5, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(opencvImage, cv2.COLOR_BGR2RGB))
    
class HexapodController: 
    def __init__(self): 
        self.walking = False

        #tripod setup
        self.tripod_a = ["forward_left", "middle_right", "back_left"]
        self.tripod_b = ["forward_right", "middle_left", "back_right"]
        self.current_tripod = self.tripod_b
        
        #default hexapod values
        self.hexa_x = 0
        self.hexa_y = 0

        self.STEP_LENGTH = 3
        self.STEP_HEIGHT = 12
        self.walking_speed = 1.5
        self.interpolation_steps = 10

        self.middlerightcoxa = 0
        self.middlerightfemur = 0
        self.middlerighttibia = 0

        self.backrightcoxa = 0
        self.backrightfemur = 0
        self.backrighttibia = 0

        self.frontrightcoxa = 0
        self.frontrightfemur = 0
        self.frontrighttibia = 0

        self.middleleftcoxa = 0
        self.middleleftfemur = 0
        self.middlelefttibia = 0

        self.backleftcoxa = 0
        self.backleftfemur = 0
        self.backlefttibia = 0

        self.frontleftcoxa = 0
        self.frontleftfemur = 0
        self.frontlefttibia = 0

        self.current_leg_positions = {
            "forward_left": (0, 0, 0),
            "forward_right": (0, 0, 0),
            "middle_left": (0, 0, 0),
            "middle_right": (0, 0, 0),
            "back_left": (0, 0, 0),
            "back_right": (0, 0, 0)
        }
        self._move_multiple_legs(self.current_leg_positions)


        #default arm values
        self.armz = 95
        self.army = 2

        self.armbase = 90
        self.armarm1 = 135
        self.armarm2 = 35
        self.armwrist1 = 127
        self.armwrist2 = 78
        self.claw = 60
        self.theta = 90

        #messager setup        
        self.message_thread = None
        self.sending_messages = False
        self.message_frequency = 10
        self.buffer_size = 1024
        self._lock = threading.Lock()
        self.connected = False
        # self.connect(HOST, PORT)

    def connect(self, host, port):
        """Establish connection to Raspberry Pi using UDP"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Changed to SOCK_DGRAM for UDP
            self.server_address = (host, port)  # Store the server address for sending
            self.connected = True
            print(f"UDP socket created for {host}:{port}")
            self.start_message_sender()  # Start sending messages
        except Exception as e:
            print(f"Socket creation error: {e}")
            self.connected = False

    def start_message_sender(self, frequency=None):
        """Start sending messages at the specified frequency (Hz)"""
        if frequency is not None:
            self.message_frequency = frequency
        
        with self._lock:
            if self.sending_messages:
                return
            self.sending_messages = True
            self.message_thread = threading.Thread(target=self._message_loop, daemon=True)
            self.message_thread.start()
    
    def stop_message_sender(self):
        """Stop sending messages"""
        with self._lock:
            self.sending_messages = False
        if self.message_thread:
            self.message_thread.join(timeout=1.0)
            self.message_thread = None

    def _message_loop(self):
        """Main loop for sending messages"""
        while self.sending_messages:
            try:
                self.send_message()
                time.sleep(1.0 / self.message_frequency)
            except Exception as e:
                print(f"Error in message sender loop: {e}")
                time.sleep(0.1)  # Brief pause on error before retrying

    def send_message(self):
        if not self.connected:
            return
        if not hasattr(self, 'seq_num'):
            self.seq_num = 0
        else:
            self.seq_num += 1
            
        message = f"<{self.seq_num}|{int(self.middlerightcoxa)},{int(self.middlerightfemur)},{int(self.middlerighttibia)},{int(self.backrightcoxa)},{int(self.backrightfemur)},{int(self.backrighttibia)},{int(self.frontrightcoxa)},{int(self.frontrightfemur)},{int(self.frontrighttibia)},{int(self.middleleftcoxa)},{int(self.middleleftfemur)},{int(self.middlelefttibia)},{int(self.backleftcoxa)},{int(self.backleftfemur)},{int(self.backlefttibia)},{int(self.frontleftcoxa)},{int(self.frontleftfemur)},{int(self.frontlefttibia)},{int(self.armbase)},{int(self.armarm1)},{int(self.armarm2)},{int(self.armwrist1)},{int(self.armwrist2)},{int(self.claw)}>\n"
        print(message)
        try:
            with self._lock:
                self.sock.sendto(message.encode(), self.server_address)  # Changed to sendto for UDP
                
        except Exception as e:
            print(f"Network error: {e}")

    def raise_tripods(self, key):
        # Define the tripods
        tripod_a = self.tripod_a  # ["forward_left", "middle_right", "back_left"]
        tripod_b = self.tripod_b  # ["forward_right", "middle_left", "back_right"]
        
        # Track the current state in the sequence
        if not hasattr(self, 'tripod_state'):
            self.tripod_state = 0
        
        # Set the height for raised legs
        raise_height = self.STEP_HEIGHT
        
        # Cycle through states: 0->1->2->3->0...
        # 0: Lift tripod A
        # 1: All legs down
        # 2: Lift tripod B
        # 3: All legs down
        if self.tripod_state == 0:
            # Raise tripod A
            targets = {leg: (0, 0, raise_height) for leg in tripod_a}
            for leg in tripod_b:
                targets[leg] = (0, 0, 0)
            self.tripod_state = 1
        elif self.tripod_state == 1:
            # All legs down
            targets = {leg: (0, 0, 0) for leg in tripod_a + tripod_b}
            self.tripod_state = 2
        elif self.tripod_state == 2:
            # Raise tripod B
            targets = {leg: (0, 0, raise_height) for leg in tripod_b}
            for leg in tripod_a:
                targets[leg] = (0, 0, 0)
            self.tripod_state = 3
        else:  # self.tripod_state == 3
            # All legs down
            targets = {leg: (0, 0, 0) for leg in tripod_a + tripod_b}
            self.tripod_state = 0
        
        # Move the legs with interpolation
        self._interpolate_movement(targets, steps=self.interpolation_steps, force=True)

    def move_arm(self, z, y, theta):
        self.armarm1, self.armarm2, self.armwrist1 = arminverse_kinematics(z, y, theta)
        print(z, y)

    def _move_multiple_legs(self, leg_positions):
        # Calculate angles for each specified leg
        for leg, (x, y, z) in leg_positions.items():
            if leg == "middle_left":
                self.middleleftcoxa, self.middleleftfemur, self.middlelefttibia = inverse_kinematics(x, y, z, "", "left")
            elif leg == "middle_right":
                self.middlerightcoxa, self.middlerightfemur, self.middlerighttibia = inverse_kinematics(x, y, z, "", "right")
            elif leg == "forward_left":
                self.frontleftcoxa, self.frontleftfemur, self.frontlefttibia = inverse_kinematics(x, y, z, "front", "left")
            elif leg == "back_left":
                self.backleftcoxa, self.backleftfemur, self.backlefttibia = inverse_kinematics(x, y, z, "back", "left")
            elif leg == "forward_right":
                self.frontrightcoxa, self.frontrightfemur, self.frontrighttibia = inverse_kinematics(x, y, z, "front", "right")
            elif leg == "back_right":
                self.backrightcoxa, self.backrightfemur, self.backrighttibia = inverse_kinematics(x, y, z, "back", "right")
            elif leg == "all":
                self.middleleftcoxa, self.middleleftfemur, self.middlelefttibia = inverse_kinematics(x, y, z, "", "left")
                self.middlerightcoxa, self.middlerightfemur, self.middlerighttibia = inverse_kinematics(x, y, z, "", "right")
                self.frontleftcoxa, self.frontleftfemur, self.frontlefttibia = inverse_kinematics(x, y, z, "front", "left")
                self.backleftcoxa, self.backleftfemur, self.backlefttibia = inverse_kinematics(x, y, z, "back", "left")
                self.frontrightcoxa, self.frontrightfemur, self.frontrighttibia = inverse_kinematics(x, y, z, "front", "right")
                self.backrightcoxa, self.backrightfemur, self.backrighttibia = inverse_kinematics(x, y, z, "back", "right")

    def _interpolate_movement(self, targets, steps, force = False):
            """Smoothly transition multiple legs to new positions over several steps"""
            if not hasattr(self, 'current_leg_positions'):
                # Initialize positions if first movement
                self.current_leg_positions = {
                    "forward_left": (0, 2, 0),
                    "forward_right": (0, 2, 0),
                    "middle_left": (0, 2, 0),
                    "middle_right": (0, 2, 0),
                    "back_left": (0, 2, 0),
                    "back_right": (0, 2, 0)
                }

            step_delay = int((100 / self.walking_speed) / steps)  # Total phase duration divided by steps

            # Calculate per-step increments for each axis
            increments = {}
            for leg, target in targets.items():
                current = self.current_leg_positions[leg]
                increments[leg] = (
                    (target[0] - current[0]) / steps,
                    (target[1] - current[1]) / steps,
                    (target[2] - current[2]) / steps
                )

            def perform_interpolation_step(remaining_steps):
                if (not force and not self.walking and not self.rotating) or remaining_steps <= 0:
                    return

                # Calculate intermediate positions
                intermediate_targets = {}
                for leg, inc in increments.items():
                    current = self.current_leg_positions[leg]
                    new_pos = (
                        round(current[0] + inc[0], 2),
                        round(current[1] + inc[1], 2),
                        round(current[2] + inc[2], 2)
                    )
                    intermediate_targets[leg] = new_pos
                    self.current_leg_positions[leg] = new_pos

                # Move to intermediate positions
                self._move_multiple_legs(intermediate_targets)
                # self.send_message()

                # Schedule next step
                if remaining_steps > 1:
                    root.after(step_delay, lambda: perform_interpolation_step(remaining_steps - 1))

            # Start interpolation sequence
            perform_interpolation_step(steps)

    def start_rotating(self, rotation):
        if self.rotating:
            return
        self.walking = False
        self.rotating = True
        self._rotate_cycle(rotation)

    def stop_rotating(self):
        self.rotating = False
        leg_positions = {leg: (0, 0, 0) for leg in self.current_leg_positions.keys()}
        self._interpolate_movement(leg_positions, steps=10, force = True)

    def _rotate_cycle(self, rotation):
        if not self.rotating:
            return
        def execute_tripod_sequence():  # Remove x,y parameters
                INTERPOLATION_STEPS = self.interpolation_steps
                first_tripod = self.tripod_a
                second_tripod = self.tripod_b
                HALF_CYCLE_DURATION = int(200 / self.walking_speed)

                def lift_tripod(tripod_group):
                    print(f"Lifting {tripod_group} tripod")
                    current_targets = {}
                    for leg in tripod_group:
                        if "back_right" in leg:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)
                        elif "forward_right" in leg:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)
                        elif "middle_right" in leg: 
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)

                    
                        elif "forward_left" in leg:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)
                        elif "back_left" in leg:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)
                        else:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)   # Back legs shorter reach


                    self._interpolate_movement(current_targets, INTERPOLATION_STEPS)
                    if self.rotating:
                        root.after(HALF_CYCLE_DURATION, lambda: swing_tripod(tripod_group))

                def swing_tripod(tripod_group):
                    print(f"Swinging {tripod_group} tripod forward")
                    # Use current self.x and self.y for direction
                    current_targets = {}
                    for leg in tripod_group:
                        if "back_right" in leg:
                            current_targets[leg] = (5, 2.5, self.STEP_HEIGHT)  # Front legs reach further
                        elif "forward_right" in leg:
                            current_targets[leg] = (5, -5, self.STEP_HEIGHT)
                        elif "middle_right" in leg: 
                            current_targets[leg] = (5, 0, self.STEP_HEIGHT)

                    
                        elif "forward_left" in leg:
                            current_targets[leg] = (-5, -2.5, self.STEP_HEIGHT)
                        elif "back_left" in leg:
                            current_targets[leg] = (-5, 5, self.STEP_HEIGHT)
                        else:
                            current_targets[leg] = (-5, 0, self.STEP_HEIGHT)   # Back legs shorter reach

                    self._interpolate_movement(current_targets, INTERPOLATION_STEPS)
                    
                    # Simultaneously reset opposing tripod on ground
                    opposing_group = second_tripod if tripod_group == first_tripod else first_tripod
                    opposing_targets = {leg: (0, 0, 0) for leg in opposing_group}
                    self._interpolate_movement(opposing_targets, INTERPOLATION_STEPS)
                    
                    if self.rotating:
                        root.after(HALF_CYCLE_DURATION, lambda: lower_tripod(tripod_group))

                def lower_tripod(tripod_group):
                    print(f"Lowering {tripod_group} tripod")
                    # Use current self.x and self.y for final position
                    current_targets = {}
                    for leg in tripod_group:
                        if "back_right" in leg:
                            current_targets[leg] = (5, 2.5, 0)  # Front legs reach further
                        elif "forward_right" in leg:
                            current_targets[leg] = (5, -5, 0)
                        elif "middle_right" in leg: 
                            current_targets[leg] = (5, 0, 0)

                    
                        elif "forward_left" in leg:
                            current_targets[leg] = (-5, -2.5, 0)
                        elif "back_left" in leg:
                            current_targets[leg] = (-5, 5, 0)
                        else:
                            current_targets[leg] = (-5, 0, 0)   # Back legs shorter reach
                    self._interpolate_movement(current_targets, INTERPOLATION_STEPS)
                    if self.rotating:
                        # Switch to opposing tripod after lowering
                        opposing_group = second_tripod if tripod_group == first_tripod else first_tripod
                        root.after(HALF_CYCLE_DURATION, lambda: lift_tripod(opposing_group))

                # Start the cycle with first tripod
                lift_tripod(first_tripod)

        def execute_left_tripod_sequence():  # Remove x,y parameters
                INTERPOLATION_STEPS = self.interpolation_steps
                HALF_CYCLE_DURATION = int(250 / self.walking_speed)
                first_tripod = ["forward_left", "middle_right", "back_left"]
                second_tripod = ["forward_right", "middle_left", "back_right"]

                def lift_tripod(tripod_group):
                    print(f"Lifting {tripod_group} tripod")
                    current_targets = {}
                    for leg in tripod_group:
                        if "back_right" in leg:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)  # Front legs reach further
                        elif "forward_right" in leg:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)
                        elif "middle_right" in leg: 
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)

                    
                        elif "forward_left" in leg:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)
                        elif "back_left" in leg:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)
                        else:
                            current_targets[leg] = (0, 0, self.STEP_HEIGHT)   # Back legs shorter reach


                    self._interpolate_movement(current_targets, INTERPOLATION_STEPS)
                    if self.rotating:
                        root.after(HALF_CYCLE_DURATION, lambda: swing_tripod(tripod_group))

                def swing_tripod(tripod_group):
                    print(f"Swinging {tripod_group} tripod forward")
                    # Use current self.x and self.y for direction
                    current_targets = {}
                    for leg in tripod_group:
                        if "forward_left" in leg:
                            current_targets[leg] = (5, 2.5, self.STEP_HEIGHT)  # Front legs reach further
                        elif "back_left" in leg:
                            current_targets[leg] = (5, -5, self.STEP_HEIGHT)
                        elif "middle_left" in leg: 
                            current_targets[leg] = (5, 0, self.STEP_HEIGHT)

                    
                        elif "back_right" in leg:
                            current_targets[leg] = (-5, -2.5, self.STEP_HEIGHT)
                        elif "forward_right" in leg:
                            current_targets[leg] = (-5, 5, self.STEP_HEIGHT)
                        else:
                            current_targets[leg] = (-5, 0, self.STEP_HEIGHT)   # Back legs shorter reach

                    self._interpolate_movement(current_targets, INTERPOLATION_STEPS)
                    
                    # Simultaneously reset opposing tripod on ground
                    opposing_group = second_tripod if tripod_group == first_tripod else first_tripod
                    opposing_targets = {leg: (0, 0, 0) for leg in opposing_group}
                    self._interpolate_movement(opposing_targets, INTERPOLATION_STEPS)
                    
                    if self.rotating:
                        root.after(HALF_CYCLE_DURATION, lambda: lower_tripod(tripod_group))

                def lower_tripod(tripod_group):
                    print(f"Lowering {tripod_group} tripod")
                    # Use current self.x and self.y for final position
                    current_targets = {}
                    for leg in tripod_group:
                        if "forward_left" in leg:
                            current_targets[leg] = (5, 2.5, 0)  # Front legs reach further
                        elif "back_left" in leg:
                            current_targets[leg] = (5, -5, 0)
                        elif "middle_left" in leg: 
                            current_targets[leg] = (5, 0, 0)

                    
                        elif "back_right" in leg:
                            current_targets[leg] = (-5, -2.5, 0)
                        elif "forward_right" in leg:
                            current_targets[leg] = (-5, 5, 0)
                        else:
                            current_targets[leg] = (-5, 0, 0)   # Back legs shorter reach
                    self._interpolate_movement(current_targets, INTERPOLATION_STEPS)
                    if self.rotating:
                        # Switch to opposing tripod after lowering
                        opposing_group = second_tripod if tripod_group == first_tripod else first_tripod
                        root.after(HALF_CYCLE_DURATION, lambda: lift_tripod(opposing_group))

                # Start the cycle with first tripod
                lift_tripod(first_tripod)

        if rotation == "right":
            execute_tripod_sequence()  # Call without parameters
        else: 
            execute_left_tripod_sequence()

    def start_walking(self):
        if self.walking:
            return
        self.walking = True
        self.rotating = False
        self._walk_cycle()

    def stop_walking(self):
        """Stop the walking cycle and reset the default stance."""
        self.walking = False
        leg_positions = {leg: (0, 0, 0) for leg in self.current_leg_positions.keys()}
        self._interpolate_movement(leg_positions, steps=10, force=True)

    def _walk_cycle(self):
        if not self.walking:
            return
        def execute_tripod_sequence():
            INTERPOLATION_STEPS = self.interpolation_steps
            HALF_CYCLE_DURATION = int(250 / self.walking_speed)

            first_tripod = self.tripod_a
            second_tripod = self.tripod_b

            def lift_tripod(tripod_group):
                print(f"Lifting {tripod_group} tripod")
                targets = {leg: (0, 0, self.STEP_HEIGHT) for leg in tripod_group}
                self._interpolate_movement(targets, INTERPOLATION_STEPS)
                
                if self.walking:
                    root.after(HALF_CYCLE_DURATION, lambda: swing_tripod(tripod_group))

            # def stance_tripod(tripod_group):
            #     opposing_group = second_tripod if tripod_group == first_tripod else first_tripod
            #     opposing_targets = {leg: (0, 0, 0) for leg in opposing_group}
            #     self._interpolate_movement(opposing_targets, INTERPOLATION_STEPS)

            #     if self.walking: 
            #         root.after(HALF_CYCLE_DURATION, lambda: swing_tripod(tripod_group))

            def swing_tripod(tripod_group):
                print(f"Swinging {tripod_group} tripod forward")
                current_targets = {
                    leg: (
                        self.hexa_x + (0 if leg in ["forward_right", "back_right", "middle_right"] else 0 if leg in ["forward_left", "back_left"] else 0), 
                        0, 
                        self.STEP_HEIGHT
                    )
                    # leg:(self.hexa_x, 0, self.STEP_HEIGHT)
                    for leg in tripod_group
                }
                self._interpolate_movement(current_targets, INTERPOLATION_STEPS)
                
                # Simultaneously reset opposing tripod on ground
                opposing_group = second_tripod if tripod_group == first_tripod else first_tripod
                opposing_targets = {leg: (0, 0, 0) for leg in opposing_group}
                self._interpolate_movement(opposing_targets, INTERPOLATION_STEPS)
                
                if self.walking:
                    root.after(HALF_CYCLE_DURATION, lambda: lower_tripod(tripod_group))

            def lower_tripod(tripod_group):
                print(f"Lowering {tripod_group} tripod")
                targets = {
                    leg: (
                        self.hexa_x + (0 if leg in ["forward_right", "back_right", "middle_right"] else 0 if leg in ["forward_left", "back_left"] else 0), 
                        0, 
                        0
                    )
                    # leg:(self.hexa_x, 0, 0)
                    for leg in tripod_group
                }
                self._interpolate_movement(targets, INTERPOLATION_STEPS)
                if self.walking:
                    # Switch to opposing tripod after lowering
                    opposing_group = second_tripod if tripod_group == first_tripod else first_tripod
                    root.after(HALF_CYCLE_DURATION, lambda: lift_tripod(opposing_group))

            # Start the cycle with first tripod
            lift_tripod(first_tripod)
        execute_tripod_sequence()  # Call without parameters


class ConnectionPanel:
    def __init__(self, root, hexapod):
        self.root = root
        self.hexapod = hexapod
        
        self.frame = ttk.Frame(root)
        self.frame.pack(pady=10)
        
        self.ip_label = ttk.Label(self.frame, text="Pi IP:")
        self.ip_label.grid(row=0, column=0)
        
        self.ip_entry = ttk.Entry(self.frame)
        self.ip_entry.insert(0, HOST)
        self.ip_entry.grid(row=0, column=1)
        
        self.connect_btn = ttk.Button(self.frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=2)

    def toggle_connection(self):
        if self.hexapod.connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        ip = self.ip_entry.get()
        self.hexapod.connect(ip, PORT)
        if self.hexapod.connected:
            self.connect_btn.config(text="Disconnect")

    def disconnect(self):
        self.hexapod.connected = False
        try:
            self.hexapod.sock.close()
        except:
            pass
        self.connect_btn.config(text="Connect")

class VirtualJoystick:
    def __init__(self, root, hexapod):
        self.root = root
        self.hexapod = hexapod
        self.step = 1
        self.max_value = 5
        self.decay_speed = 0.8
        self.active_keys = set()
        
        self.root.bind('<KeyPress>', self._on_key_press)
        self.root.bind('<KeyRelease>', self._on_key_release)
        self._update()

    def _on_key_press(self, event):
        keysym = event.keysym.lower()
        if keysym in ['w', 'a', 's', 'd', 'q', 'e', 'i', 'k', 'j', 'l', 'up', 'down','left','right', 'o','p' ,'return', 'h']:
            self.active_keys.add(keysym)
            if keysym in ['q', 'e']:
                rotation = 'left' if keysym == 'e' else 'right'
                self.hexapod.start_rotating(rotation)
            if keysym in ['return']:
                self.hexapod.claw = 145 if self.hexapod.claw == 60 else 60
                # self.hexapod.send_message()
            if keysym == 'h':
                self.hexapod.raise_tripods('space')

    def _on_key_release(self, event):
        keysym = event.keysym.lower()
        if keysym in ['w', 'a', 's', 'd', 'q', 'e', 'i', 'j', 'k', 'l', 'up', 'down','left','right','o','p','return']:
            self.active_keys.discard(keysym)
            if keysym in ['q', 'e']:
                self.hexapod.stop_rotating()

    def _update(self):
        if 'w' in self.active_keys:
            self.hexapod.hexa_x = hexapod.STEP_LENGTH
        elif 's' in self.active_keys:
            self.hexapod.hexa_x = -hexapod.STEP_LENGTH
        else:
            self.hexapod.hexa_x = 0 
            
        if 'a' in self.active_keys:
            self.hexapod.hexa_y = hexapod.STEP_LENGTH
        elif 'd' in self.active_keys:
            self.hexapod.hexa_y = -hexapod.STEP_LENGTH
        else:
            self.hexapod.hexa_y = 0

        arm_step = 3
        if 'o' in self.active_keys or 'p' in self.active_keys or 'j' in self.active_keys or 'l' in self.active_keys or 'i' in self.active_keys or 'k' in self.active_keys or 'up' in self.active_keys or 'down' in self.active_keys or 'left' in self.active_keys or 'right' in self.active_keys:
            if 'j' in self.active_keys:
                self.hexapod.armbase = max(0, self.hexapod.armbase + arm_step/3)
            if 'l' in self.active_keys:
                self.hexapod.armbase = min(180, self.hexapod.armbase - arm_step/3)
            
            if 'p' in self.active_keys:
                self.hexapod.armz = min(300, self.hexapod.armz + arm_step)
            if 'o' in self.active_keys:
                self.hexapod.armz = max(0, self.hexapod.armz - arm_step)
            # Vertical arm movement (I/K)
            if 'i' in self.active_keys:
                self.hexapod.army = min(200, self.hexapod.army + arm_step)
            if 'k' in self.active_keys:
                self.hexapod.army = max(-200, self.hexapod.army - arm_step)

            if 'up' in self.active_keys:
                self.hexapod.theta = min(180, self.hexapod.theta + arm_step)
            if 'down' in self.active_keys:
                self.hexapod.theta = max(0, self.hexapod.theta - arm_step)

            if 'left' in self.active_keys:
                self.hexapod.armwrist2 = min(180, self.hexapod.armwrist2 - arm_step)
            if 'right' in self.active_keys:
                self.hexapod.armwrist2 = max(0, self.hexapod.armwrist2 + arm_step)
            try:
                self.hexapod.move_arm(self.hexapod.armz, self.hexapod.army, self.hexapod.theta)
                # self.hexapod.send_message()
            except ValueError as e:
                print(f"Arm movement error: {e}")

        # Auto-start/stop walking based on movement keys
        movement_keys_active = any(k in self.active_keys for k in ['w', 'a', 's', 'd'])
        
        if movement_keys_active and not self.hexapod.walking:
            self.hexapod.start_walking()
        elif not movement_keys_active and self.hexapod.walking:
            self.hexapod.stop_walking()

        self.root.after(16, self._update)











hexapod = HexapodController()
video_stream = VideoStream(root, HOST)
hexapod.start_message_sender(frequency=10) 
ConnectionPanel(root, hexapod)
root.title("Hexapod Controller")
VirtualJoystick(root, hexapod)


def on_closing():
        hexapod.stop_message_sender()
        if hexapod.connected:
            hexapod.sock.close()
        root.destroy()


root.mainloop()