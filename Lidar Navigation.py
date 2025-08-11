"""
Enhanced RPLidar A1 Object Detection and Mapping GUI
Features: Real-time SLAM mapping, object tracking, comprehensive GUI controls
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from collections import deque
import json
import pickle
from datetime import datetime
import cv2
from scipy.spatial.distance import cdist
from rplidar import RPLidar
import serial.tools.list_ports

class EnhancedRPLidarGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced RPLidar Scanner with Real-time Mapping")
        self.root.geometry("1400x900")
        
        # Core components
        self.lidar = None
        self.running = False
        self.scan_thread = None
        self.connected = False
        
        # Data storage
        self.scan_data = deque(maxlen=1000)
        self.current_objects = []
        self.map_points = deque(maxlen=5000)  # Global map points
        self.object_history = deque(maxlen=100)  # Object tracking history
        self.trajectory = []  # Robot trajectory
        
        # Pose estimation (SLAM-like)
        self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}  # Current lidar position
        self.previous_scan = None
        self.pose_history = deque(maxlen=1000)
        self.enable_pose_estimation = tk.BooleanVar(value=True)
        
        # Detection parameters
        self.params = {
            'min_distance': tk.DoubleVar(value=0.1),
            'max_distance': tk.DoubleVar(value=6.0),
            'min_quality': tk.IntVar(value=10),
            'cluster_distance': tk.DoubleVar(value=0.3),
            'min_cluster_points': tk.IntVar(value=3),
            'map_decay': tk.DoubleVar(value=0.95),  # Map point decay factor
            'object_persistence': tk.IntVar(value=5),  # Frames to keep objects
            'pose_estimation_threshold': tk.DoubleVar(value=0.3),  # ICP threshold
            'max_pose_change': tk.DoubleVar(value=1.0)  # Max pose change per frame (m)
        }
        
        # GUI state
        self.show_map = tk.BooleanVar(value=True)
        self.show_objects = tk.BooleanVar(value=True)
        self.show_trajectory = tk.BooleanVar(value=True)
        self.auto_save = tk.BooleanVar(value=False)
        self.recording = False
        
        # Statistics
        self.stats = {
            'scans_processed': 0,
            'objects_detected': 0,
            'map_points': 0,
            'scan_rate': 0
        }
        
        self.setup_gui()
        self.setup_plot()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)
        
        # Right panel for plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(control_frame)
        self.setup_plot_panel(plot_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel with all settings"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Connection tab
        conn_frame = ttk.Frame(notebook)
        notebook.add(conn_frame, text="Connection")
        self.setup_connection_tab(conn_frame)
        
        # Detection tab
        detect_frame = ttk.Frame(notebook)
        notebook.add(detect_frame, text="Detection")
        self.setup_detection_tab(detect_frame)
        
        # Mapping tab
        map_frame = ttk.Frame(notebook)
        notebook.add(map_frame, text="Mapping")
        self.setup_mapping_tab(map_frame)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        self.setup_statistics_tab(stats_frame)
        
    def setup_connection_tab(self, parent):
        """Setup connection controls"""
        # Port selection
        ttk.Label(parent, text="Serial Port:").pack(anchor=tk.W, pady=2)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(parent, textvariable=self.port_var, state="readonly")
        self.port_combo.pack(fill=tk.X, pady=2)
        
        ttk.Button(parent, text="Refresh Ports", command=self.refresh_ports).pack(fill=tk.X, pady=2)
        
        # Baudrate
        ttk.Label(parent, text="Baudrate:").pack(anchor=tk.W, pady=(10,2))
        self.baudrate_var = tk.IntVar(value=115200)
        baudrate_combo = ttk.Combobox(parent, textvariable=self.baudrate_var, 
                                     values=[9600, 19200, 38400, 57600, 115200, 256000])
        baudrate_combo.pack(fill=tk.X, pady=2)
        baudrate_combo.set("115200")
        
        # Connection buttons
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.connect_btn = ttk.Button(parent, text="Connect", command=self.connect_lidar)
        self.connect_btn.pack(fill=tk.X, pady=2)
        
        self.scan_btn = ttk.Button(parent, text="Start Scanning", 
                                  command=self.toggle_scanning, state=tk.DISABLED)
        self.scan_btn.pack(fill=tk.X, pady=2)
        
        # Status
        ttk.Label(parent, text="Status:").pack(anchor=tk.W, pady=(10,2))
        self.status_label = ttk.Label(parent, text="Disconnected", foreground="red")
        self.status_label.pack(anchor=tk.W, pady=2)
        
        # Initialize ports
        self.refresh_ports()
        
    def setup_detection_tab(self, parent):
        """Setup detection parameter controls"""
        # Distance range
        ttk.Label(parent, text="Distance Range (m):").pack(anchor=tk.W, pady=2)
        
        frame1 = ttk.Frame(parent)
        frame1.pack(fill=tk.X, pady=2)
        ttk.Label(frame1, text="Min:").pack(side=tk.LEFT)
        ttk.Scale(frame1, from_=0.05, to=1.0, variable=self.params['min_distance'],
                 orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame1, textvariable=self.params['min_distance']).pack(side=tk.RIGHT)
        
        frame2 = ttk.Frame(parent)
        frame2.pack(fill=tk.X, pady=2)
        ttk.Label(frame2, text="Max:").pack(side=tk.LEFT)
        ttk.Scale(frame2, from_=1.0, to=12.0, variable=self.params['max_distance'],
                 orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame2, textvariable=self.params['max_distance']).pack(side=tk.RIGHT)
        
        # Quality threshold
        ttk.Label(parent, text="Min Quality:").pack(anchor=tk.W, pady=(10,2))
        frame3 = ttk.Frame(parent)
        frame3.pack(fill=tk.X, pady=2)
        ttk.Scale(frame3, from_=0, to=100, variable=self.params['min_quality'],
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame3, textvariable=self.params['min_quality']).pack(side=tk.RIGHT)
        
        # Clustering parameters
        ttk.Label(parent, text="Cluster Distance (m):").pack(anchor=tk.W, pady=(10,2))
        frame4 = ttk.Frame(parent)
        frame4.pack(fill=tk.X, pady=2)
        ttk.Scale(frame4, from_=0.1, to=1.0, variable=self.params['cluster_distance'],
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame4, textvariable=self.params['cluster_distance']).pack(side=tk.RIGHT)
        
        ttk.Label(parent, text="Min Cluster Points:").pack(anchor=tk.W, pady=(10,2))
        frame5 = ttk.Frame(parent)
        frame5.pack(fill=tk.X, pady=2)
        ttk.Scale(frame5, from_=2, to=20, variable=self.params['min_cluster_points'],
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame5, textvariable=self.params['min_cluster_points']).pack(side=tk.RIGHT)
        
    def setup_mapping_tab(self, parent):
        """Setup mapping controls"""
        # Display options
        ttk.Label(parent, text="Display Options:").pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(parent, text="Show Map", variable=self.show_map).pack(anchor=tk.W)
        ttk.Checkbutton(parent, text="Show Objects", variable=self.show_objects).pack(anchor=tk.W)
        ttk.Checkbutton(parent, text="Show Trajectory", variable=self.show_trajectory).pack(anchor=tk.W)
        ttk.Checkbutton(parent, text="Enable Pose Estimation", variable=self.enable_pose_estimation).pack(anchor=tk.W)
        
        # Pose estimation parameters
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(parent, text="Pose Estimation:").pack(anchor=tk.W, pady=2)
        
        frame_pose1 = ttk.Frame(parent)
        frame_pose1.pack(fill=tk.X, pady=2)
        ttk.Label(frame_pose1, text="ICP Threshold:").pack(side=tk.LEFT)
        ttk.Scale(frame_pose1, from_=0.1, to=1.0, variable=self.params['pose_estimation_threshold'],
                 orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame_pose1, textvariable=self.params['pose_estimation_threshold']).pack(side=tk.RIGHT)
        
        frame_pose2 = ttk.Frame(parent)
        frame_pose2.pack(fill=tk.X, pady=2)
        ttk.Label(frame_pose2, text="Max Pose Change:").pack(side=tk.LEFT)
        ttk.Scale(frame_pose2, from_=0.1, to=2.0, variable=self.params['max_pose_change'],
                 orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame_pose2, textvariable=self.params['max_pose_change']).pack(side=tk.RIGHT)
        
        # Current pose display
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(parent, text="Current Pose:").pack(anchor=tk.W, pady=2)
        self.pose_label = ttk.Label(parent, text="X: 0.00, Y: 0.00, θ: 0.0°")
        self.pose_label.pack(anchor=tk.W, pady=2)
        
        # Map parameters
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="Map Decay Factor:").pack(anchor=tk.W, pady=2)
        frame1 = ttk.Frame(parent)
        frame1.pack(fill=tk.X, pady=2)
        ttk.Scale(frame1, from_=0.8, to=1.0, variable=self.params['map_decay'],
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame1, textvariable=self.params['map_decay']).pack(side=tk.RIGHT)
        
        # Map controls
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Button(parent, text="Clear Map", command=self.clear_map).pack(fill=tk.X, pady=2)
        ttk.Button(parent, text="Reset Pose", command=self.reset_pose).pack(fill=tk.X, pady=2)
        ttk.Button(parent, text="Save Map", command=self.save_map).pack(fill=tk.X, pady=2)
        ttk.Button(parent, text="Load Map", command=self.load_map).pack(fill=tk.X, pady=2)
        
        # Recording
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        self.record_btn = ttk.Button(parent, text="Start Recording", command=self.toggle_recording)
        self.record_btn.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(parent, text="Auto-save scans", variable=self.auto_save).pack(anchor=tk.W, pady=2)
        
    def setup_statistics_tab(self, parent):
        """Setup statistics display"""
        self.stats_labels = {}
        
        for key in self.stats.keys():
            ttk.Label(parent, text=f"{key.replace('_', ' ').title()}:").pack(anchor=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(parent, text="0")
            self.stats_labels[key].pack(anchor=tk.W, pady=2)
            
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Button(parent, text="Reset Statistics", command=self.reset_statistics).pack(fill=tk.X, pady=2)
        
    def setup_plot_panel(self, parent):
        """Setup the matplotlib plot panel"""
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('white')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        
    def setup_plot(self):
        """Initialize the plot"""
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Real-time RPLidar Mapping and Object Detection')
        self.ax.axis('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)
        
        # Animation
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        
    def refresh_ports(self):
        """Refresh available COM ports"""
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        self.port_combo['values'] = port_list
        if port_list:
            self.port_combo.set(port_list[0])
            
    def connect_lidar(self):
        """Connect to RPLidar"""
        if self.connected:
            self.disconnect_lidar()
            return
            
        port = self.port_var.get()
        baudrate = self.baudrate_var.get()
        
        if not port:
            messagebox.showerror("Error", "Please select a port")
            return
            
        try:
            self.lidar = RPLidar(port, baudrate=baudrate, timeout=3)
            info = self.lidar.get_info()
            health = self.lidar.get_health()
            
            self.connected = True
            self.status_label.config(text="Connected", foreground="green")
            self.connect_btn.config(text="Disconnect")
            self.scan_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("Success", f"Connected to RPLidar\nInfo: {info}\nHealth: {health}")
            
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")
            
    def disconnect_lidar(self):
        """Disconnect from RPLidar"""
        try:
            if self.running:
                self.stop_scanning()
            if self.lidar:
                self.lidar.disconnect()
            self.connected = False
            self.status_label.config(text="Disconnected", foreground="red")
            self.connect_btn.config(text="Connect")
            self.scan_btn.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Disconnect error: {e}")
            
    def toggle_scanning(self):
        """Start/stop scanning"""
        if self.running:
            self.stop_scanning()
        else:
            self.start_scanning()
            
    def start_scanning(self):
        """Start continuous scanning"""
        if not self.connected:
            messagebox.showerror("Error", "Not connected to lidar")
            return
            
        try:
            self.running = True
            self.lidar.start_motor()
            time.sleep(2)
            
            self.scan_thread = threading.Thread(target=self.scan_loop)
            self.scan_thread.daemon = True
            self.scan_thread.start()
            
            self.scan_btn.config(text="Stop Scanning")
            
        except Exception as e:
            messagebox.showerror("Scan Error", f"Failed to start scanning: {str(e)}")
            self.running = False
            
    def stop_scanning(self):
        """Stop scanning"""
        self.running = False
        if self.scan_thread:
            self.scan_thread.join(timeout=2)
        if self.lidar:
            self.lidar.stop()
            self.lidar.stop_motor()
        self.scan_btn.config(text="Start Scanning")
        
    def scan_loop(self):
        """Main scanning loop"""
        scan_count = 0
        start_time = time.time()
        
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=500):
                if not self.running:
                    break
                    
                scan_points = self.process_scan(scan)
                self.scan_data.append(scan_points)
                
                # Estimate pose change if enabled
                if self.enable_pose_estimation.get() and self.previous_scan:
                    pose_change = self.estimate_pose_change(self.previous_scan, scan_points)
                    self.update_pose(pose_change)
                
                # Transform points to global coordinates
                global_points = self.transform_to_global(scan_points)
                
                # Update map with global coordinates
                self.update_map(global_points)
                
                # Detect objects in global coordinates
                self.current_objects = self.detect_objects(global_points)
                self.update_object_history()
                
                # Store current scan for next iteration
                self.previous_scan = scan_points.copy()
                
                # Update statistics
                scan_count += 1
                self.stats['scans_processed'] = scan_count
                self.stats['objects_detected'] = len(self.current_objects)
                self.stats['map_points'] = len(self.map_points)
                
                # Calculate scan rate
                if scan_count % 10 == 0:
                    elapsed = time.time() - start_time
                    self.stats['scan_rate'] = round(scan_count / elapsed, 1)
                
                # Auto-save if enabled
                if self.auto_save.get() and scan_count % 100 == 0:
                    self.auto_save_data()
                    
        except Exception as e:
            print(f"Scan loop error: {e}")
            
    def process_scan(self, scan):
        """Process raw scan data"""
        scan_points = []
        for quality, angle, distance in scan:
            if (quality > self.params['min_quality'].get() and 
                self.params['min_distance'].get() <= distance/1000.0 <= self.params['max_distance'].get()):
                
                dist_m = distance / 1000.0
                x = dist_m * math.cos(math.radians(angle))
                y = dist_m * math.sin(math.radians(angle))
                scan_points.append((x, y, dist_m, angle, quality))
        
        return scan_points
    
    def estimate_pose_change(self, prev_scan, curr_scan):
        """Estimate pose change using simplified ICP-like algorithm"""
        if not prev_scan or not curr_scan:
            return {'dx': 0, 'dy': 0, 'dtheta': 0}
        
        # Convert to numpy arrays for easier processing
        prev_points = np.array([(p[0], p[1]) for p in prev_scan])
        curr_points = np.array([(p[0], p[1]) for p in curr_scan])
        
        if len(prev_points) < 10 or len(curr_points) < 10:
            return {'dx': 0, 'dy': 0, 'dtheta': 0}
        
        # Simple feature matching - find correspondences
        correspondences = self.find_correspondences(prev_points, curr_points)
        
        if len(correspondences) < 5:
            return {'dx': 0, 'dy': 0, 'dtheta': 0}
        
        # Calculate transformation
        prev_matched = np.array([prev_points[i] for i, j in correspondences])
        curr_matched = np.array([curr_points[j] for i, j in correspondences])
        
        # Calculate centroids
        prev_centroid = np.mean(prev_matched, axis=0)
        curr_centroid = np.mean(curr_matched, axis=0)
        
        # Simple translation estimate
        dx = curr_centroid[0] - prev_centroid[0]
        dy = curr_centroid[1] - prev_centroid[1]
        
        # Simple rotation estimate (simplified)
        dtheta = self.estimate_rotation(prev_matched - prev_centroid, 
                                       curr_matched - curr_centroid)
        
        # Limit maximum change to prevent jumps
        max_change = self.params['max_pose_change'].get()
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > max_change:
            scale = max_change / distance
            dx *= scale
            dy *= scale
        
        if abs(dtheta) > math.radians(30):  # Limit to 30 degrees
            dtheta = math.radians(30) * (1 if dtheta > 0 else -1)
        
        return {'dx': -dx, 'dy': -dy, 'dtheta': -dtheta}  # Invert for lidar movement
    
    def find_correspondences(self, prev_points, curr_points, max_distance=0.3):
        """Find point correspondences between scans"""
        correspondences = []
        threshold = self.params['pose_estimation_threshold'].get()
        
        # Use distance matrix to find nearest neighbors
        if len(prev_points) > 100 or len(curr_points) > 100:
            # Subsample for performance
            prev_sample = prev_points[::max(1, len(prev_points)//100)]
            curr_sample = curr_points[::max(1, len(curr_points)//100)]
        else:
            prev_sample = prev_points
            curr_sample = curr_points
        
        distances = cdist(prev_sample, curr_sample)
        
        for i in range(len(prev_sample)):
            j = np.argmin(distances[i])
            if distances[i, j] < threshold:
                correspondences.append((i, j))
        
        return correspondences
    
    def estimate_rotation(self, prev_points, curr_points):
        """Estimate rotation between point sets"""
        if len(prev_points) < 3 or len(curr_points) < 3:
            return 0
        
        # Calculate cross-correlation matrix
        H = np.dot(prev_points.T, curr_points) / len(prev_points)
        
        # Use SVD to find rotation
        try:
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            
            # Extract rotation angle
            angle = math.atan2(R[1, 0], R[0, 0])
            return angle
        except:
            return 0
    
    def update_pose(self, pose_change):
        """Update current pose based on estimated change"""
        # Apply rotation first, then translation
        cos_theta = math.cos(self.current_pose['theta'])
        sin_theta = math.sin(self.current_pose['theta'])
        
        # Transform translation to global coordinates
        global_dx = pose_change['dx'] * cos_theta - pose_change['dy'] * sin_theta
        global_dy = pose_change['dx'] * sin_theta + pose_change['dy'] * cos_theta
        
        # Update pose
        self.current_pose['x'] += global_dx
        self.current_pose['y'] += global_dy
        self.current_pose['theta'] += pose_change['dtheta']
        
        # Normalize angle
        while self.current_pose['theta'] > math.pi:
            self.current_pose['theta'] -= 2 * math.pi
        while self.current_pose['theta'] < -math.pi:
            self.current_pose['theta'] += 2 * math.pi
        
        # Add to trajectory
        self.trajectory.append({
            'x': self.current_pose['x'],
            'y': self.current_pose['y'],
            'theta': self.current_pose['theta'],
            'timestamp': time.time()
        })
        
        # Limit trajectory length
        if len(self.trajectory) > 1000:
            self.trajectory.pop(0)
        
        # Store pose history
        self.pose_history.append(self.current_pose.copy())
    
    def transform_to_global(self, scan_points):
        """Transform scan points to global coordinate system"""
        if not scan_points:
            return []
        
        global_points = []
        cos_theta = math.cos(self.current_pose['theta'])
        sin_theta = math.sin(self.current_pose['theta'])
        
        for point in scan_points:
            local_x, local_y = point[0], point[1]
            
            # Rotate and translate to global coordinates
            global_x = (local_x * cos_theta - local_y * sin_theta) + self.current_pose['x']
            global_y = (local_x * sin_theta + local_y * cos_theta) + self.current_pose['y']
            
            # Keep additional information
            global_point = (global_x, global_y, point[2], point[3], point[4])
            global_points.append(global_point)
        
        return global_points
    
    def update_map(self, scan_points):
        """Update the global map with new scan points"""
        decay_factor = self.params['map_decay'].get()
        
        # Decay existing map points
        for i in range(len(self.map_points)):
            if len(self.map_points[i]) > 5:  # Has confidence value
                self.map_points[i] = (*self.map_points[i][:5], 
                                    self.map_points[i][5] * decay_factor)
        
        # Add new points
        for point in scan_points:
            x, y, dist, angle, quality = point
            # Add confidence based on quality
            confidence = min(quality / 100.0, 1.0)
            map_point = (x, y, dist, angle, quality, confidence)
            self.map_points.append(map_point)
            
        # Remove points with very low confidence
        self.map_points = deque([p for p in self.map_points if p[5] > 0.1], 
                               maxlen=5000)
    
    def detect_objects(self, scan_points):
        """Enhanced object detection with tracking"""
        if not scan_points:
            return []
        
        points = sorted(scan_points, key=lambda p: math.atan2(p[1], p[0]))
        objects = []
        current_cluster = []
        cluster_distance = self.params['cluster_distance'].get()
        min_points = self.params['min_cluster_points'].get()
        
        for point in points:
            if not current_cluster:
                current_cluster = [point]
            else:
                last_point = current_cluster[-1]
                distance = math.sqrt((point[0] - last_point[0])**2 + 
                                   (point[1] - last_point[1])**2)
                
                if distance <= cluster_distance:
                    current_cluster.append(point)
                else:
                    if len(current_cluster) >= min_points:
                        objects.append(self.create_object_info(current_cluster))
                    current_cluster = [point]
        
        if len(current_cluster) >= min_points:
            objects.append(self.create_object_info(current_cluster))
        
        return objects
    
    def create_object_info(self, cluster_points):
        """Create enhanced object information"""
        x_coords = [p[0] for p in cluster_points]
        y_coords = [p[1] for p in cluster_points]
        qualities = [p[4] for p in cluster_points]
        
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        distance = math.sqrt(centroid_x**2 + centroid_y**2)
        angle = math.degrees(math.atan2(centroid_y, centroid_x))
        
        # Enhanced measurements
        width = math.sqrt((max(x_coords) - min(x_coords))**2 + 
                         (max(y_coords) - min(y_coords))**2)
        avg_quality = sum(qualities) / len(qualities)
        
        return {
            'points': cluster_points,
            'centroid': (centroid_x, centroid_y),
            'distance': distance,
            'angle': angle,
            'width': width,
            'point_count': len(cluster_points),
            'avg_quality': avg_quality,
            'timestamp': time.time(),
            'id': None  # Will be assigned during tracking
        }
    
    def update_object_history(self):
        """Update object tracking history"""
        # Simple object tracking based on proximity
        for obj in self.current_objects:
            closest_match = None
            min_distance = float('inf')
            
            # Look for closest match in recent history
            for hist_obj in reversed(list(self.object_history)[-10:]):
                for old_obj in hist_obj:
                    if old_obj.get('id') is not None:
                        dist = math.sqrt((obj['centroid'][0] - old_obj['centroid'][0])**2 +
                                       (obj['centroid'][1] - old_obj['centroid'][1])**2)
                        if dist < min_distance and dist < 0.5:  # 50cm threshold
                            min_distance = dist
                            closest_match = old_obj
            
            if closest_match:
                obj['id'] = closest_match['id']
            else:
                obj['id'] = len([o for hist in self.object_history for o in hist]) + len(self.current_objects)
        
        self.object_history.append(self.current_objects.copy())
    
    def update_plot(self, frame):
        """Update the matplotlib plot"""
        self.ax.clear()
        
        # Plot map points if enabled
        if self.show_map.get() and self.map_points:
            map_x = [p[0] for p in self.map_points]
            map_y = [p[1] for p in self.map_points]
            map_conf = [p[5] for p in self.map_points]
            
            scatter = self.ax.scatter(map_x, map_y, c=map_conf, s=1, 
                                    alpha=0.6, cmap='viridis', 
                                    vmin=0, vmax=1, label='Map Points')
        
        # Plot current scan points (in global coordinates)
        if self.scan_data:
            latest_scan = list(self.scan_data)[-1]
            if latest_scan:
                # Transform latest scan to global coordinates
                global_scan = self.transform_to_global(latest_scan)
                scan_x = [p[0] for p in global_scan]
                scan_y = [p[1] for p in global_scan]
                self.ax.scatter(scan_x, scan_y, c='lightblue', s=2, 
                              alpha=0.8, label='Current Scan')
        
        # Plot trajectory if enabled
        if self.show_trajectory.get() and len(self.trajectory) > 1:
            traj_x = [p['x'] for p in self.trajectory]
            traj_y = [p['y'] for p in self.trajectory]
            self.ax.plot(traj_x, traj_y, 'g-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # Plot detected objects if enabled
        if self.show_objects.get() and self.current_objects:
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.current_objects)))
            
            for i, (obj, color) in enumerate(zip(self.current_objects, colors)):
                obj_x = [p[0] for p in obj['points']]
                obj_y = [p[1] for p in obj['points']]
                
                self.ax.scatter(obj_x, obj_y, c=[color], s=20, alpha=0.8,
                              label=f'Object {obj.get("id", i+1)}')
                
                # Mark centroid
                self.ax.scatter(obj['centroid'][0], obj['centroid'][1],
                              c=[color], s=100, marker='x', linewidths=3)
                
                # Add object info text
                self.ax.annotate(f'ID:{obj.get("id", i+1)}\n{obj["distance"]:.2f}m',
                               obj['centroid'], xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        # Plot current lidar position and orientation
        lidar_x, lidar_y = self.current_pose['x'], self.current_pose['y']
        lidar_theta = self.current_pose['theta']
        
        # Lidar position
        self.ax.scatter([lidar_x], [lidar_y], c='red', s=150, marker='o', 
                       label='Lidar Position', zorder=10)
        
        # Lidar orientation arrow
        arrow_length = 0.5
        arrow_dx = arrow_length * math.cos(lidar_theta)
        arrow_dy = arrow_length * math.sin(lidar_theta)
        self.ax.arrow(lidar_x, lidar_y, arrow_dx, arrow_dy, 
                     head_width=0.1, head_length=0.1, fc='red', ec='red', zorder=10)
        
        # Styling
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title(f'Enhanced RPLidar Mapping ({len(self.current_objects)} objects, '
                         f'{len(self.map_points)} map points)')
        self.ax.axis('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)
        
        # Update statistics display
        self.update_statistics_display()
        
        plt.tight_layout()
    
    def update_statistics_display(self):
        """Update statistics labels"""
        for key, label in self.stats_labels.items():
            label.config(text=str(self.stats[key]))
        
        # Update pose display
        if hasattr(self, 'pose_label'):
            pose_text = f"X: {self.current_pose['x']:.2f}, Y: {self.current_pose['y']:.2f}, θ: {math.degrees(self.current_pose['theta']):.1f}°"
            self.pose_label.config(text=pose_text)
    
    def reset_pose(self):
        """Reset lidar pose to origin"""
        self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.trajectory.clear()
        self.pose_history.clear()
        self.previous_scan = None
        messagebox.showinfo("Info", "Pose reset to origin")
    
    def clear_map(self):
        """Clear the current map"""
        self.map_points.clear()
        self.object_history.clear()
        # Don't clear pose/trajectory - user might want to keep moving
        messagebox.showinfo("Info", "Map cleared")
    
    def save_map(self):
        """Save current map to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                data = {
                    'map_points': list(self.map_points),
                    'object_history': list(self.object_history),
                    'trajectory': self.trajectory,
                    'pose_history': list(self.pose_history),
                    'current_pose': self.current_pose,
                    'timestamp': datetime.now().isoformat(),
                    'stats': self.stats.copy()
                }
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
                messagebox.showinfo("Success", f"Map saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save map: {str(e)}")
    
    def load_map(self):
        """Load map from file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                
                self.map_points = deque(data['map_points'], maxlen=5000)
                if 'object_history' in data:
                    self.object_history = deque(data['object_history'], maxlen=100)
                if 'trajectory' in data:
                    self.trajectory = data['trajectory']
                if 'pose_history' in data:
                    self.pose_history = deque(data['pose_history'], maxlen=1000)
                if 'current_pose' in data:
                    self.current_pose = data['current_pose']
                
                messagebox.showinfo("Success", f"Map loaded from {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load map: {str(e)}")
    
    def toggle_recording(self):
        """Toggle scan recording"""
        if self.recording:
            self.recording = False
            self.record_btn.config(text="Start Recording")
        else:
            self.recording = True
            self.record_btn.config(text="Stop Recording")
    
    def auto_save_data(self):
        """Auto-save scan data"""
        if not self.recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rplidar_scan_{timestamp}.json"
        
        try:
            data = {
                'timestamp': timestamp,
                'scan_count': len(self.scan_data),
                'recent_scans': [list(scan) for scan in list(self.scan_data)[-10:]],
                'current_objects': self.current_objects,
                'stats': self.stats.copy()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Auto-save error: {e}")
    
    def reset_statistics(self):
        """Reset all statistics"""
        for key in self.stats:
            self.stats[key] = 0
        messagebox.showinfo("Info", "Statistics reset")
    
    def on_closing(self):
        """Handle application closing"""
        if self.running:
            self.stop_scanning()
        if self.connected:
            self.disconnect_lidar()
        self.root.destroy()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = EnhancedRPLidarGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()