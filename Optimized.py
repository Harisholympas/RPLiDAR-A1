import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from collections import deque
import pickle
from datetime import datetime
from rplidar import RPLidar
import serial.tools.list_ports

class EnhancedRPLidarGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced RPLidar Scanner")
        self.root.geometry("1200x800")
        
        self.lidar = None
        self.running = False
        self.connected = False
        
        # Use numpy arrays for better performance
        self.scan_data = deque(maxlen=500)
        self.map_points = deque(maxlen=10000)
        self.object_history = deque(maxlen=50)
        self.trajectory = deque(maxlen=500)
        
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.previous_scan = None
        self.enable_pose_estimation = tk.BooleanVar(value=True)
        
        self.params = {
            'min_distance': tk.DoubleVar(value=0.1),
            'max_distance': tk.DoubleVar(value=6.0),
            'min_quality': tk.IntVar(value=10),
            'cluster_distance': tk.DoubleVar(value=0.3),
            'min_cluster_points': tk.IntVar(value=3),
            'map_decay': tk.DoubleVar(value=0.95),
            'pose_threshold': tk.DoubleVar(value=0.3),
            'max_pose_change': tk.DoubleVar(value=0.5)
        }
        
        self.display_options = {
            'show_map': tk.BooleanVar(value=True),
            'show_objects': tk.BooleanVar(value=True),
            'show_trajectory': tk.BooleanVar(value=True)
        }
        
        self.stats = {
            'scans_processed': 0,
            'objects_detected': 0,
            'map_points': 0,
            'scan_rate': 0.0
        }
        
        self.setup_gui()
        self.setup_plot()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        control_frame = ttk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        control_frame.pack_propagate(False)
        
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self._setup_control_panel(control_frame)
        self._setup_plot_panel(plot_frame)
        
    def _setup_control_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        tabs = [
            ("Connection", self._setup_connection_tab),
            ("Detection", self._setup_detection_tab),
            ("Mapping", self._setup_mapping_tab),
            ("Stats", self._setup_stats_tab)
        ]
        
        for tab_name, setup_func in tabs:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)
            setup_func(frame)
        
    def _setup_connection_tab(self, parent):
        ttk.Label(parent, text="Serial Port:").pack(anchor=tk.W, pady=2)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(parent, textvariable=self.port_var, state="readonly")
        self.port_combo.pack(fill=tk.X, pady=2)
        
        ttk.Button(parent, text="Refresh Ports", command=self._refresh_ports).pack(fill=tk.X, pady=2)
        
        ttk.Label(parent, text="Baudrate:").pack(anchor=tk.W, pady=(10,2))
        self.baudrate_var = tk.IntVar(value=115200)
        ttk.Combobox(parent, textvariable=self.baudrate_var, 
                    values=[115200, 256000], state="readonly").pack(fill=tk.X, pady=2)
        
        ttk.Separator(parent).pack(fill=tk.X, pady=10)
        self.connect_btn = ttk.Button(parent, text="Connect", command=self._toggle_connection)
        self.connect_btn.pack(fill=tk.X, pady=2)
        
        self.scan_btn = ttk.Button(parent, text="Start Scan", 
                                 command=self._toggle_scanning, state=tk.DISABLED)
        self.scan_btn.pack(fill=tk.X, pady=2)
        
        ttk.Label(parent, text="Status:").pack(anchor=tk.W, pady=(10,2))
        self.status_label = ttk.Label(parent, text="Disconnected", foreground="red")
        self.status_label.pack(anchor=tk.W)
        
        self._refresh_ports()
        
    def _setup_detection_tab(self, parent):
        params = [
            ("Distance Range (m)", [
                ("Min:", 'min_distance', 0.05, 1.0),
                ("Max:", 'max_distance', 1.0, 12.0)
            ]),
            ("Quality", [
                ("Min Quality:", 'min_quality', 0, 100)
            ]),
            ("Clustering", [
                ("Cluster Distance (m):", 'cluster_distance', 0.1, 1.0),
                ("Min Points:", 'min_cluster_points', 2, 20)
            ])
        ]
        
        for section, controls in params:
            ttk.Label(parent, text=section).pack(anchor=tk.W, pady=(10,2))
            for label, key, min_val, max_val in controls:
                frame = ttk.Frame(parent)
                frame.pack(fill=tk.X, pady=2)
                ttk.Label(frame, text=label).pack(side=tk.LEFT)
                ttk.Scale(frame, from_=min_val, to=max_val, variable=self.params[key],
                         orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
                ttk.Label(frame, textvariable=self.params[key]).pack(side=tk.RIGHT)
        
    def _setup_mapping_tab(self, parent):
        for label, var in self.display_options.items():
            ttk.Checkbutton(parent, text=label.replace('_', ' ').title(), variable=var).pack(anchor=tk.W)
        
        ttk.Separator(parent).pack(fill=tk.X, pady=5)
        ttk.Label(parent, text="Pose Estimation:").pack(anchor=tk.W, pady=2)
        
        pose_params = [
            ("ICP Threshold:", 'pose_threshold', 0.1, 1.0),
            ("Max Pose Change:", 'max_pose_change', 0.1, 2.0)
        ]
        
        for label, key, min_val, max_val in pose_params:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label).pack(side=tk.LEFT)
            ttk.Scale(frame, from_=min_val, to=max_val, variable=self.params[key],
                     orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(frame, textvariable=self.params[key]).pack(side=tk.RIGHT)
        
        ttk.Separator(parent).pack(fill=tk.X, pady=5)
        ttk.Label(parent, text="Current Pose:").pack(anchor=tk.W)
        self.pose_label = ttk.Label(parent, text="X: 0.00, Y: 0.00, θ: 0.0°")
        self.pose_label.pack(anchor=tk.W, pady=2)
        
        ttk.Separator(parent).pack(fill=tk.X, pady=5)
        ttk.Label(parent, text="Map Decay:").pack(anchor=tk.W, pady=2)
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Scale(frame, from_=0.8, to=1.0, variable=self.params['map_decay'],
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame, textvariable=self.params['map_decay']).pack(side=tk.RIGHT)
        
        buttons = [
            ("Clear Map", self._clear_map),
            ("Reset Pose", self._reset_pose),
            ("Save Map", self._save_map),
            ("Load Map", self._load_map)
        ]
        
        for text, command in buttons:
            ttk.Button(parent, text=text, command=command).pack(fill=tk.X, pady=2)
        
    def _setup_stats_tab(self, parent):
        self.stats_labels = {}
        for key in self.stats:
            ttk.Label(parent, text=f"{key.replace('_', ' ').title()}:").pack(anchor=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(parent, text="0")
            self.stats_labels[key].pack(anchor=tk.W, pady=2)
        
        ttk.Button(parent, text="Reset Stats", command=self._reset_stats).pack(fill=tk.X, pady=10)
        
    def _setup_plot_panel(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_plot(self):
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('RPLidar Mapping')
        self.ax.axis('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        
        self.animation = FuncAnimation(self.fig, self._update_plot, interval=100, 
                                     blit=False, cache_frame_data=False)
        
    def _refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.set(ports[0])
            
    def _toggle_connection(self):
        if self.connected:
            self._disconnect_lidar()
        else:
            self._connect_lidar()
            
    def _connect_lidar(self):
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Error", "Select a port")
            return
            
        try:
            self.lidar = RPLidar(port, baudrate=self.baudrate_var.get(), timeout=1)
            self.lidar.get_info()
            self.connected = True
            self.status_label.config(text="Connected", foreground="green")
            self.connect_btn.config(text="Disconnect")
            self.scan_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")
            
    def _disconnect_lidar(self):
        if self.running:
            self._stop_scanning()
        if self.lidar:
            self.lidar.disconnect()
        self.connected = False
        self.status_label.config(text="Disconnected", foreground="red")
        self.connect_btn.config(text="Connect")
        self.scan_btn.config(state=tk.DISABLED)
        
    def _toggle_scanning(self):
        if self.running:
            self._stop_scanning()
        else:
            self._start_scanning()
            
    def _start_scanning(self):
        if not self.connected:
            return
            
        try:
            self.running = True
            self.lidar.start_motor()
            time.sleep(1)
            self.scan_thread = threading.Thread(target=self._scan_loop)
            self.scan_thread.daemon = True
            self.scan_thread.start()
            self.scan_btn.config(text="Stop Scan")
        except Exception as e:
            messagebox.showerror("Error", f"Scan failed: {str(e)}")
            self.running = False
            
    def _stop_scanning(self):
        self.running = False
        if hasattr(self, 'scan_thread'):
            self.scan_thread.join(timeout=1)
        if self.lidar:
            self.lidar.stop()
            self.lidar.stop_motor()
        self.scan_btn.config(text="Start Scan")
        
    def _scan_loop(self):
        start_time = time.time()
        scan_count = 0
        
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=500):
                if not self.running:
                    break
                    
                scan_points = self._process_scan(scan)
                self.scan_data.append(scan_points)
                
                if self.enable_pose_estimation.get() and self.previous_scan is not None:
                    pose_change = self._estimate_pose_change(self.previous_scan, scan_points)
                    self._update_pose(pose_change)
                
                global_points = self._transform_to_global(scan_points)
                self._update_map(global_points)
                self.current_objects = self._detect_objects(global_points)
                
                scan_count += 1
                self.stats['scans_processed'] = scan_count
                self.stats['objects_detected'] = len(self.current_objects)
                self.stats['map_points'] = len(self.map_points)
                self.stats['scan_rate'] = scan_count / (time.time() - start_time)
                
                self.previous_scan = scan_points
        except Exception as e:
            print(f"Scan error: {e}")
            
    def _process_scan(self, scan):
        scan_points = np.array([
            (d/1000.0 * math.cos(math.radians(a)), 
             d/1000.0 * math.sin(math.radians(a)))
            for q, a, d in scan
            if q > self.params['min_quality'].get() and 
               self.params['min_distance'].get() <= d/1000.0 <= self.params['max_distance'].get()
        ])
        return scan_points if scan_points.size else np.empty((0, 2))
    
    def _estimate_pose_change(self, prev_scan, curr_scan):
        if not prev_scan.size or not curr_scan.size:
            return np.zeros(3)
            
        # Subsample points for performance
        sample_size = 100
        prev_sample = prev_scan[::max(1, len(prev_scan)//sample_size)]
        curr_sample = curr_scan[::max(1, len(curr_scan)//sample_size)]
        
        if len(prev_sample) < 10 or len(curr_sample) < 10:
            return np.zeros(3)
            
        # Fast nearest neighbor search
        distances = np.min(np.sum((prev_sample[:, None] - curr_sample)**2, axis=2), axis=1)
        matches = distances < self.params['pose_threshold'].get()
        
        if np.sum(matches) < 5:
            return np.zeros(3)
            
        prev_matched = prev_sample[matches]
        curr_matched = curr_sample[np.argmin(np.sum((prev_sample[:, None] - curr_sample)**2, axis=2), axis=1)[matches]]
        
        # Calculate transformation
        prev_centroid = np.mean(prev_matched, axis=0)
        curr_centroid = np.mean(curr_matched, axis=0)
        
        dx, dy = curr_centroid - prev_centroid
        H = np.dot((prev_matched - prev_centroid).T, (curr_matched - curr_centroid))
        U, _, Vt = np.linalg.svd(H)
        dtheta = math.atan2(np.dot(Vt.T, U.T)[1, 0], np.dot(Vt.T, U.T)[0, 0])
        
        # Limit pose change
        max_change = self.params['max_pose_change'].get()
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > max_change:
            scale = max_change / dist
            dx *= scale
            dy *= scale
        dtheta = np.clip(dtheta, -np.pi/6, np.pi/6)
        
        return np.array([-dx, -dy, -dtheta])
    
    def _update_pose(self, pose_change):
        cos_theta = math.cos(self.current_pose[2])
        sin_theta = math.sin(self.current_pose[2])
        
        global_dx = pose_change[0] * cos_theta - pose_change[1] * sin_theta
        global_dy = pose_change[0] * sin_theta + pose_change[1] * cos_theta
        
        self.current_pose += np.array([global_dx, global_dy, pose_change[2]])
        self.current_pose[2] = (self.current_pose[2] + np.pi) % (2 * np.pi) - np.pi
        self.trajectory.append(self.current_pose.copy())
        
    def _transform_to_global(self, scan_points):
        if not scan_points.size:
            return np.empty((0, 2))
            
        cos_theta = math.cos(self.current_pose[2])
        sin_theta = math.sin(self.current_pose[2])
        rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        return np.dot(scan_points, rotation.T) + self.current_pose[:2]
    
    def _update_map(self, global_points):
        if not global_points.size:
            return
            
        decay = self.params['map_decay'].get()
        new_points = np.column_stack((global_points, np.ones(len(global_points))))
        
        if self.map_points:
            existing = np.array(self.map_points)
            existing[:, 2] *= decay
            mask = existing[:, 2] > 0.1
            self.map_points = deque(existing[mask], maxlen=10000)
        
        self.map_points.extend(new_points)
        
    def _detect_objects(self, points):
        if not points.size:
            return []
            
        points = points[np.argsort(np.arctan2(points[:, 1], points[:, 0]))]
        cluster_dist = self.params['cluster_distance'].get()
        min_points = self.params['min_cluster_points'].get()
        
        objects = []
        current_cluster = []
        
        for i, point in enumerate(points):
            if not current_cluster:
                current_cluster = [point]
            elif np.sqrt(np.sum((point - current_cluster[-1])**2)) <= cluster_dist:
                current_cluster.append(point)
            else:
                if len(current_cluster) >= min_points:
                    objects.append(self._create_object_info(current_cluster))
                current_cluster = [point]
                
        if len(current_cluster) >= min_points:
            objects.append(self._create_object_info(current_cluster))
            
        return objects
    
    def _create_object_info(self, points):
        centroid = np.mean(points, axis=0)
        distance = np.sqrt(np.sum(centroid**2))
        angle = np.degrees(np.arctan2(centroid[1], centroid[0]))
        width = np.max(np.ptp(points, axis=0))
        
        return {
            'points': points,
            'centroid': centroid,
            'distance': distance,
            'angle': angle,
            'width': width,
            'point_count': len(points),
            'timestamp': time.time()
        }
    
    def _update_plot(self, frame):
        self.ax.clear()
        
        if self.display_options['show_map'].get() and self.map_points:
            points = np.array(self.map_points)
            self.ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                          s=1, alpha=0.6, cmap='viridis', vmin=0, vmax=1)
        
        if self.scan_data and self.display_options['show_objects'].get():
            scan_points = self._transform_to_global(list(self.scan_data)[-1])
            if scan_points.size:
                self.ax.scatter(scan_points[:, 0], scan_points[:, 1], 
                              c='lightblue', s=2, alpha=0.8)
        
        if self.display_options['show_trajectory'].get() and len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'g-', linewidth=2, alpha=0.7)
        
        for i, obj in enumerate(self.current_objects):
            self.ax.scatter(obj['points'][:, 0], obj['points'][:, 1], 
                          c=f'C{i}', s=20, alpha=0.8)
            self.ax.scatter(obj['centroid'][0], obj['centroid'][1], 
                          c=f'C{i}', s=100, marker='x')
            self.ax.annotate(f'ID:{i+1}\n{obj["distance"]:.2f}m',
                           obj['centroid'], xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        self.ax.scatter([self.current_pose[0]], [self.current_pose[1]], 
                       c='red', s=100, marker='o')
        arrow_len = 0.3
        self.ax.arrow(self.current_pose[0], self.current_pose[1],
                     arrow_len * math.cos(self.current_pose[2]),
                     arrow_len * math.sin(self.current_pose[2]),
                     head_width=0.1, fc='red')
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f'RPLidar Map ({len(self.current_objects)} objects)')
        self.ax.axis('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        
        self._update_stats_display()
        
    def _update_stats_display(self):
        for key, label in self.stats_labels.items():
            label.config(text=f"{self.stats[key]:.1f}" if key == 'scan_rate' else str(self.stats[key]))
        
        self.pose_label.config(
            text=f"X: {self.current_pose[0]:.2f}, Y: {self.current_pose[1]:.2f}, "
                 f"θ: {np.degrees(self.current_pose[2]):.1f}°"
        )
        
    def _clear_map(self):
        self.map_points.clear()
        self.object_history.clear()
        self.trajectory.clear()
        
    def _reset_pose(self):
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.previous_scan = None
        
    def _save_map(self):
        filename = filedialog.asksaveasfilename(defaultextension=".pkl", 
                                              filetypes=[("Pickle files", "*.pkl")])
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'map_points': list(self.map_points),
                    'object_history': list(self.object_history),
                    'trajectory': list(self.trajectory),
                    'current_pose': self.current_pose,
                    'timestamp': datetime.now().isoformat()
                }, f)
                
    def _load_map(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if filename:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.map_points = deque(data.get('map_points', []), maxlen=10000)
                self.object_history = deque(data.get('object_history', []), maxlen=50)
                self.trajectory = deque(data.get('trajectory', []), maxlen=500)
                self.current_pose = data.get('current_pose', np.array([0.0, 0.0, 0.0]))
                
    def _reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0
            
    def on_closing(self):
        if self.running:
            self._stop_scanning()
        if self.connected:
            self._disconnect_lidar()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = EnhancedRPLidarGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()