"""
Optimized RPLidar A1 Real-time SLAM Mapping GUI
Features: Real-time SLAM mapping, pose estimation, comprehensive GUI controls
Optimized for performance with object detection removed
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
from scipy.spatial.distance import cdist
from rplidar import RPLidar
import serial.tools.list_ports

class OptimizedRPLidarGUI:
    __slots__ = ['root', 'lidar', 'running', 'scan_thread', 'connected', 
                 'scan_data', 'map_points', 'trajectory', 'current_pose', 
                 'previous_scan', 'pose_history', 'enable_pose_estimation',
                 'params', 'show_map', 'show_trajectory', 'auto_save', 
                 'recording', 'stats', 'fig', 'ax', 'canvas', 'animation',
                 'port_var', 'port_combo', 'baudrate_var', 'connect_btn',
                 'scan_btn', 'status_label', 'pose_label', 'stats_labels',
                 'record_btn', '_last_update_time', '_scan_count_cache',
                 '_performance_counter']
    
    def __init__(self, root):
        self.root = root
        self.root.title("Optimized RPLidar Scanner with Real-time Mapping")
        self.root.geometry("1400x900")
        
        # Core variables
        self.lidar = None
        self.running = False
        self.scan_thread = None
        self.connected = False
        
        # Data containers - optimized sizes
        self.scan_data = deque(maxlen=500)  # Reduced from 1000
        self.map_points = deque(maxlen=3000)  # Reduced from 5000
        self.trajectory = []
        
        # Pose tracking
        self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.previous_scan = None
        self.pose_history = deque(maxlen=500)  # Reduced from 1000
        self.enable_pose_estimation = tk.BooleanVar(value=True)
        
        # Parameters - using direct values for frequently accessed ones
        self.params = {
            'min_distance': tk.DoubleVar(value=0.1),
            'max_distance': tk.DoubleVar(value=6.0),
            'min_quality': tk.IntVar(value=10),
            'map_decay': tk.DoubleVar(value=0.95),
            'pose_estimation_threshold': tk.DoubleVar(value=0.3),
            'max_pose_change': tk.DoubleVar(value=1.0)
        }
        
        # Display options
        self.show_map = tk.BooleanVar(value=True)
        self.show_trajectory = tk.BooleanVar(value=True)
        self.auto_save = tk.BooleanVar(value=False)
        self.recording = False
        
        # Statistics
        self.stats = {
            'scans_processed': 0,
            'map_points': 0,
            'scan_rate': 0
        }
        
        # Performance optimization variables
        self._last_update_time = 0
        self._scan_count_cache = 0
        self._performance_counter = 0
        
        self.setup_gui()
        self.setup_plot()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)
        
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
        
        # Detection tab (simplified)
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
        
        self.refresh_ports()
        
    def setup_detection_tab(self, parent):
        """Setup simplified detection parameter controls"""
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
        
    def setup_mapping_tab(self, parent):
        """Setup mapping controls"""
        # Display options
        ttk.Label(parent, text="Display Options:").pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(parent, text="Show Map", variable=self.show_map).pack(anchor=tk.W)
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
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        
    def setup_plot(self):
        """Initialize the plot"""
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Optimized RPLidar Real-time Mapping')
        self.ax.axis('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)
        
        # Reduced animation interval for better performance
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=150, blit=False)
        
    def refresh_ports(self):
        """Refresh available COM ports"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.set(ports[0])
            
    def connect_lidar(self):
        """Connect to RPLidar"""
        if self.connected:
            self.disconnect_lidar()
            return
            
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Error", "Please select a port")
            return
            
        try:
            self.lidar = RPLidar(port, baudrate=self.baudrate_var.get(), timeout=3)
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
            
            self.scan_thread = threading.Thread(target=self.scan_loop, daemon=True)
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
        """Optimized main scanning loop"""
        scan_count = 0
        start_time = time.time()
        
        # Cache frequently accessed values
        min_dist = self.params['min_distance'].get()
        max_dist = self.params['max_distance'].get()
        min_qual = self.params['min_quality'].get()
        enable_pose = self.enable_pose_estimation.get()
        
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=800):  # Reduced buffer
                if not self.running:
                    break
                    
                # Process scan with cached values
                scan_points = self._process_scan_optimized(scan, min_dist, max_dist, min_qual)
                self.scan_data.append(scan_points)
                
                # Pose estimation (only if enabled and have previous scan)
                if enable_pose and self.previous_scan and len(scan_points) > 10:
                    pose_change = self._estimate_pose_change_optimized(self.previous_scan, scan_points)
                    self._update_pose_optimized(pose_change)
                
                # Update map with optimized method
                self._update_map_optimized(self._transform_to_global_optimized(scan_points))
                
                self.previous_scan = scan_points
                
                # Update statistics less frequently
                scan_count += 1
                if scan_count % 5 == 0:  # Update every 5 scans instead of every scan
                    self.stats['scans_processed'] = scan_count
                    self.stats['map_points'] = len(self.map_points)
                    
                    # Calculate scan rate every 20 scans
                    if scan_count % 20 == 0:
                        elapsed = time.time() - start_time
                        self.stats['scan_rate'] = round(scan_count / elapsed, 1)
                
                # Auto-save less frequently
                if self.auto_save.get() and scan_count % 200 == 0:  # Reduced frequency
                    self.auto_save_data()
                    
        except Exception as e:
            print(f"Scan loop error: {e}")
    
    def _process_scan_optimized(self, scan, min_dist, max_dist, min_qual):
        """Optimized scan processing using list comprehension and cached values"""
        return [(distance * 0.001 * math.cos(math.radians(angle)),
                distance * 0.001 * math.sin(math.radians(angle)),
                distance * 0.001, angle, quality)
                for quality, angle, distance in scan
                if quality > min_qual and min_dist <= distance * 0.001 <= max_dist]
    
    def _estimate_pose_change_optimized(self, prev_scan, curr_scan):
        """Optimized pose estimation with reduced complexity"""
        if len(prev_scan) < 10 or len(curr_scan) < 10:
            return {'dx': 0, 'dy': 0, 'dtheta': 0}
        
        # Subsample for performance (every 3rd point)
        prev_points = np.array([(p[0], p[1]) for p in prev_scan[::3]])
        curr_points = np.array([(p[0], p[1]) for p in curr_scan[::3]])
        
        if len(prev_points) < 5 or len(curr_points) < 5:
            return {'dx': 0, 'dy': 0, 'dtheta': 0}
        
        # Simplified correspondence finding
        correspondences = self._find_correspondences_optimized(prev_points, curr_points)
        
        if len(correspondences) < 3:
            return {'dx': 0, 'dy': 0, 'dtheta': 0}
        
        # Calculate transformation with cached values
        prev_matched = prev_points[correspondences[:, 0]]
        curr_matched = curr_points[correspondences[:, 1]]
        
        prev_centroid = np.mean(prev_matched, axis=0)
        curr_centroid = np.mean(curr_matched, axis=0)
        
        dx = curr_centroid[0] - prev_centroid[0]
        dy = curr_centroid[1] - prev_centroid[1]
        
        # Simplified rotation estimation
        dtheta = self._estimate_rotation_optimized(prev_matched - prev_centroid, 
                                                  curr_matched - curr_centroid)
        
        # Apply limits
        max_change = self.params['max_pose_change'].get()
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > max_change:
            scale = max_change / distance
            dx *= scale
            dy *= scale
        
        dtheta = max(-math.radians(30), min(math.radians(30), dtheta))
        
        return {'dx': -dx, 'dy': -dy, 'dtheta': -dtheta}
    
    def _find_correspondences_optimized(self, prev_points, curr_points):
        """Optimized correspondence finding"""
        if len(prev_points) > 50 or len(curr_points) > 50:
            # Further subsample for very large point sets
            prev_sample = prev_points[::max(1, len(prev_points)//50)]
            curr_sample = curr_points[::max(1, len(curr_points)//50)]
        else:
            prev_sample = prev_points
            curr_sample = curr_points
        
        distances = cdist(prev_sample, curr_sample)
        threshold = self.params['pose_estimation_threshold'].get()
        
        # Vectorized correspondence finding
        min_indices = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(distances)), min_indices]
        valid_mask = min_distances < threshold
        
        valid_prev_indices = np.where(valid_mask)[0]
        valid_curr_indices = min_indices[valid_mask]
        
        return np.column_stack((valid_prev_indices, valid_curr_indices))
    
    def _estimate_rotation_optimized(self, prev_points, curr_points):
        """Optimized rotation estimation"""
        if len(prev_points) < 3:
            return 0
        
        try:
            H = np.dot(prev_points.T, curr_points) / len(prev_points)
            U, _, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            return math.atan2(R[1, 0], R[0, 0])
        except:
            return 0
    
    def _update_pose_optimized(self, pose_change):
        """Optimized pose update"""
        cos_theta = math.cos(self.current_pose['theta'])
        sin_theta = math.sin(self.current_pose['theta'])
        
        global_dx = pose_change['dx'] * cos_theta - pose_change['dy'] * sin_theta
        global_dy = pose_change['dx'] * sin_theta + pose_change['dy'] * cos_theta
        
        self.current_pose['x'] += global_dx
        self.current_pose['y'] += global_dy
        self.current_pose['theta'] += pose_change['dtheta']
        
        # Normalize angle
        while self.current_pose['theta'] > math.pi:
            self.current_pose['theta'] -= 2 * math.pi
        while self.current_pose['theta'] < -math.pi:
            self.current_pose['theta'] += 2 * math.pi
        
        # Add to trajectory (limit frequency)
        self._performance_counter += 1
        if self._performance_counter % 3 == 0:  # Every 3rd pose update
            self.trajectory.append({
                'x': self.current_pose['x'],
                'y': self.current_pose['y'],
                'theta': self.current_pose['theta'],
                'timestamp': time.time()
            })
            
            if len(self.trajectory) > 500:  # Reduced from 1000
                self.trajectory.pop(0)
        
        self.pose_history.append(self.current_pose.copy())
    
    def _transform_to_global_optimized(self, scan_points):
        """Optimized global coordinate transformation"""
        if not scan_points:
            return []
        
        cos_theta = math.cos(self.current_pose['theta'])
        sin_theta = math.sin(self.current_pose['theta'])
        pos_x = self.current_pose['x']
        pos_y = self.current_pose['y']
        
        return [(local_x * cos_theta - local_y * sin_theta + pos_x,
                local_x * sin_theta + local_y * cos_theta + pos_y,
                point[2], point[3], point[4])
                for local_x, local_y, *point in zip(*zip(*scan_points))]
    
    def _update_map_optimized(self, scan_points):
        """Optimized map update with batch processing"""
        if not scan_points:
            return
            
        decay_factor = self.params['map_decay'].get()
        
        # Batch decay existing points
        if self.map_points:
            for i in range(len(self.map_points)):
                point = self.map_points[i]
                if len(point) > 5:
                    self.map_points[i] = (*point[:5], point[5] * decay_factor)
        
        # Add new points with confidence
        for point in scan_points:
            confidence = min(point[4] / 100.0, 1.0)  # point[4] is quality
            map_point = (*point, confidence)
            self.map_points.append(map_point)
        
        # Remove low confidence points less frequently
        if self._performance_counter % 10 == 0:
            self.map_points = deque([p for p in self.map_points if p[5] > 0.1], 
                                   maxlen=3000)
    
    def update_plot(self, frame):
        """Optimized plot update with reduced frequency"""
        current_time = time.time()
        if current_time - self._last_update_time < 0.1:  # Limit to 10 FPS
            return
        self._last_update_time = current_time
        
        self.ax.clear()
        
        # Plot map points if enabled
        if self.show_map.get() and self.map_points:
            # Subsample for display performance
            step = max(1, len(self.map_points) // 2000)  # Max 2000 points
            sampled_points = list(self.map_points)[::step]
            
            if sampled_points:
                map_x = [p[0] for p in sampled_points]
                map_y = [p[1] for p in sampled_points]
                map_conf = [p[5] for p in sampled_points]
                
                self.ax.scatter(map_x, map_y, c=map_conf, s=1, 
                              alpha=0.6, cmap='viridis', 
                              vmin=0, vmax=1, label='Map Points')
        
        # Plot current scan
        if self.scan_data:
            latest_scan = list(self.scan_data)[-1]
            if latest_scan:
                global_scan = self._transform_to_global_optimized(latest_scan)
                if global_scan:
                    scan_x = [p[0] for p in global_scan]
                    scan_y = [p[1] for p in global_scan]
                    self.ax.scatter(scan_x, scan_y, c='lightblue', s=2, 
                                  alpha=0.8, label='Current Scan')
        
        # Plot trajectory
        if self.show_trajectory.get() and len(self.trajectory) > 1:
            # Subsample trajectory for performance
            step = max(1, len(self.trajectory) // 200)  # Max 200 trajectory points
            sampled_traj = self.trajectory[::step]
            
            traj_x = [p['x'] for p in sampled_traj]
            traj_y = [p['y'] for p in sampled_traj]
            self.ax.plot(traj_x, traj_y, 'g-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # Plot lidar position and orientation
        lidar_x, lidar_y = self.current_pose['x'], self.current_pose['y']
        lidar_theta = self.current_pose['theta']
        
        self.ax.scatter([lidar_x], [lidar_y], c='red', s=150, marker='o', 
                       label='Lidar Position', zorder=10)
        
        arrow_length = 0.5
        arrow_dx = arrow_length * math.cos(lidar_theta)
        arrow_dy = arrow_length * math.sin(lidar_theta)
        self.ax.arrow(lidar_x, lidar_y, arrow_dx, arrow_dy, 
                     head_width=0.1, head_length=0.1, fc='red', ec='red', zorder=10)
        
        # Set plot properties
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title(f'Optimized RPLidar Mapping ({len(self.map_points)} map points)')
        self.ax.axis('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)
        
        # Update statistics display less frequently
        if self._performance_counter % 5 == 0:
            self.update_statistics_display()
        
        plt.tight_layout()
    
    def update_statistics_display(self):
        """Update statistics labels"""
        for key, label in self.stats_labels.items():
            label.config(text=str(self.stats[key]))
        
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
                
                self.map_points = deque(data['map_points'], maxlen=3000)
                if 'trajectory' in data:
                    self.trajectory = data['trajectory']
                if 'pose_history' in data:
                    self.pose_history = deque(data['pose_history'], maxlen=500)
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
            # Save only essential data for performance
            data = {
                'timestamp': timestamp,
                'scan_count': len(self.scan_data),
                'recent_scans': [list(scan) for scan in list(self.scan_data)[-5:]],  # Reduced from 10
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
        self._scan_count_cache = 0
        self._performance_counter = 0
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
    app = OptimizedRPLidarGUI(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()