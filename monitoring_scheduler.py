# monitoring_scheduler.py
"""
Standalone service for scheduled medicine monitoring
Run this as a separate service for 24/7 monitoring capabilities
"""

import time
import schedule
import logging
import json
import cv2
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from enhanced_detect_mediapipe_gesture import EnhancedMedicineDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicineMonitoringScheduler:
    def __init__(self, config_file: str = "monitoring_config.json"):
        self.config_file = config_file
        self.detector = EnhancedMedicineDetector()
        self.config = self.load_config()
        self.monitoring_history = []
        self.is_running = False
        
        # Email notification settings
        self.email_config = self.config.get("email_notifications", {})
        
        logger.info("Medicine Monitoring Scheduler initialized")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            "scheduled_times": [
                {"time": "09:00", "duration": 60, "enabled": True},
                {"time": "21:00", "duration": 60, "enabled": True}
            ],
            "monitoring_settings": {
                "max_missed_days": 2,
                "alert_caregivers": True,
                "save_video_clips": False,
                "detection_sensitivity": "medium"
            },
            "email_notifications": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            },
            "storage": {
                "results_directory": "monitoring_results",
                "video_clips_directory": "monitoring_clips",
                "max_storage_days": 30
            }
        }
        
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    config = {**default_config, **loaded_config}
                    logger.info(f"Configuration loaded from {self.config_file}")
                    return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            # Create default config file
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config
        else:
            self.config = config
            
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def setup_directories(self):
        """Create necessary directories"""
        storage_config = self.config.get("storage", {})
        
        results_dir = Path(storage_config.get("results_directory", "monitoring_results"))
        clips_dir = Path(storage_config.get("video_clips_directory", "monitoring_clips"))
        
        results_dir.mkdir(exist_ok=True)
        clips_dir.mkdir(exist_ok=True)
        
        logger.info(f"Directories created: {results_dir}, {clips_dir}")
    
    def scheduled_monitoring_job(self, schedule_config: Dict[str, Any]):
        """Execute a scheduled monitoring session"""
        if not schedule_config.get("enabled", True):
            logger.info(f"Monitoring at {schedule_config['time']} is disabled")
            return
        
        duration = schedule_config.get("duration", 60)
        logger.info(f"Starting scheduled monitoring session: {duration}s at {datetime.now()}")
        
        try:
            # Start monitoring
            monitoring_result = self.run_monitoring_session(duration)
            
            # Save results
            self.save_monitoring_result(monitoring_result, schedule_config)
            
            # Check for alerts
            self.check_and_send_alerts(monitoring_result)
            
            # Update history
            self.monitoring_history.append({
                "timestamp": datetime.now().isoformat(),
                "scheduled_time": schedule_config["time"],
                "duration": duration,
                "result": monitoring_result
            })
            
            # Keep history manageable (last 100 sessions)
            if len(self.monitoring_history) > 100:
                self.monitoring_history = self.monitoring_history[-100:]
            
            logger.info(f"Monitoring session completed: {'SUCCESS' if monitoring_result.get('medicine_intake_detected') else 'NO_DETECTION'}")
            
        except Exception as e:
            logger.error(f"Monitoring session failed: {e}")
            self.send_error_notification(str(e))
    
    def run_monitoring_session(self, duration: int) -> Dict[str, Any]:
        """Run a single monitoring session"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not access webcam")
        
        detection_events = []
        start_time = time.time()
        frame_count = 0
        
        # Optional: Record video if enabled
        video_writer = None
        video_filename = None
        if self.config.get("monitoring_settings", {}).get("save_video_clips", False):
            clips_dir = Path(self.config.get("storage", {}).get("video_clips_directory", "monitoring_clips"))
            video_filename = clips_dir / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_writer = cv2.VideoWriter(str(video_filename), fourcc, fps, (width, height))
        
        try:
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    continue
                
                frame_count += 1
                
                # Save frame if recording
                if video_writer:
                    video_writer.write(frame)
                
                # Perform detection every 10 frames for efficiency
                if frame_count % 10 == 0:
                    detection = self.detector.detect_medicine_intake_gesture(frame)
                    
                    # Check for medicine intake
                    is_medicine_intake = (
                        detection["hand_near_mouth"] and 
                        detection["head_tilted_back"] and
                        detection["medicine_object_detected"]
                    )
                    
                    if is_medicine_intake:
                        event = {
                            "timestamp": datetime.now().isoformat(),
                            "frame_number": frame_count,
                            "detection_details": detection
                        }
                        detection_events.append(event)
                        logger.info("Medicine intake detected during scheduled monitoring!")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
        
        # Prepare result
        result = {
            "monitoring_date": datetime.now().isoformat(),
            "duration_seconds": duration,
            "medicine_intake_detected": len(detection_events) > 0,
            "detection_events": detection_events,
            "total_detections": len(detection_events),
            "total_frames": frame_count,
            "video_file": str(video_filename) if video_filename else None
        }
        
        return result
    
    def save_monitoring_result(self, result: Dict[str, Any], schedule_config: Dict[str, Any]):
        """Save monitoring result to file"""
        results_dir = Path(self.config.get("storage", {}).get("results_directory", "monitoring_results"))
        
        # Add schedule info to result
        result_with_schedule = {
            **result,
            "scheduled_time": schedule_config["time"],
            "schedule_config": schedule_config
        }
        
        filename = f"monitoring_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result_with_schedule, f, indent=2)
            logger.info(f"Monitoring result saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save monitoring result: {e}")
    
    def check_and_send_alerts(self, monitoring_result: Dict[str, Any]):
        """Check if alerts should be sent based on monitoring result"""
        if not self.config.get("monitoring_settings", {}).get("alert_caregivers", True):
            return
        
        # Check for missed medicine
        if not monitoring_result.get("medicine_intake_detected", False):
            self.check_missed_medicine_alerts()
        
        # Send success notification if configured
        if monitoring_result.get("medicine_intake_detected", False):
            self.send_success_notification(monitoring_result)
    
    def check_missed_medicine_alerts(self):
        """Check for consecutive missed medicine and send alerts"""
        max_missed_days = self.config.get("monitoring_settings", {}).get("max_missed_days", 2)
        
        # Count recent missed sessions
        recent_history = [h for h in self.monitoring_history[-10:]]  # Check last 10 sessions
        consecutive_misses = 0
        
        for session in reversed(recent_history):
            if not session.get("result", {}).get("medicine_intake_detected", False):
                consecutive_misses += 1
            else:
                break
        
        if consecutive_misses >= max_missed_days:
            self.send_missed_medicine_alert(consecutive_misses)
    
    def send_success_notification(self, result: Dict[str, Any]):
        """Send success notification"""
        if not self.email_config.get("enabled", False):
            return
        
        subject = "✅ Medicine Intake Confirmed"
        message = f"""
Medicine intake successfully detected!

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Detections: {result.get('total_detections', 0)}

Keep up the good work with your medication schedule!
        """
        
        self.send_email_notification(subject, message)
    
    def send_missed_medicine_alert(self, consecutive_misses: int):
        """Send missed medicine alert"""
        if not self.email_config.get("enabled", False):
            return
        
        subject = f"⚠️ Medicine Intake Alert - {consecutive_misses} Missed Sessions"
        message = f"""
ALERT: Medicine intake not detected for {consecutive_misses} consecutive scheduled sessions.

Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check on the patient and ensure they are taking their medication as prescribed.

If this is a false alarm (patient took medicine but wasn't detected), please review:
- Camera positioning and lighting
- Medicine strip/bottle visibility
- Hand-to-mouth gesture clarity
        """
        
        self.send_email_notification(subject, message)
    
    def send_error_notification(self, error_message: str):
        """Send error notification"""
        if not self.email_config.get("enabled", False):
            return
        
        subject = "❌ Medicine Monitoring System Error"
        message = f"""
The medicine monitoring system encountered an error:

Error: {error_message}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the system and restart if necessary.
        """
        
        self.send_email_notification(subject, message)
    
    def send_email_notification(self, subject: str, message: str):
        """Send email notification"""
        try:
            smtp_server = self.email_config.get("smtp_server", "smtp.gmail.com")
            smtp_port = self.email_config.get("smtp_port", 587)
            username = self.email_config.get("username", "")
            password = self.email_config.get("password", "")
            recipients = self.email_config.get("recipients", [])
            
            if not username or not password or not recipients:
                logger.warning("Email configuration incomplete, skipping notification")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(username, recipients, text)
            server.quit()
            
            logger.info(f"Email notification sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def setup_scheduled_jobs(self):
        """Setup all scheduled monitoring jobs"""
        schedule.clear()  # Clear existing jobs
        
        scheduled_times = self.config.get("scheduled_times", [])
        
        for schedule_config in scheduled_times:
            if schedule_config.get("enabled", True):
                time_str = schedule_config["time"]
                
                # Create a closure to capture the schedule_config
                def create_job(config):
                    return lambda: self.scheduled_monitoring_job(config)
                
                schedule.every().day.at(time_str).do(create_job(schedule_config))
                logger.info(f"Scheduled monitoring job: {time_str} for {schedule_config.get('duration', 60)}s")
    
    def cleanup_old_files(self):
        """Clean up old monitoring files"""
        max_days = self.config.get("storage", {}).get("max_storage_days", 30)
        cutoff_date = datetime.now() - timedelta(days=max_days)
        
        # Clean up result files
        results_dir = Path(self.config.get("storage", {}).get("results_directory", "monitoring_results"))
        if results_dir.exists():
            deleted_count = 0
            for file_path in results_dir.glob("monitoring_result_*.json"):
                try:
                    if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} old result files")
        
        # Clean up video clips
        clips_dir = Path(self.config.get("storage", {}).get("video_clips_directory", "monitoring_clips"))
        if clips_dir.exists():
            deleted_count = 0
            for file_path in clips_dir.glob("monitoring_*.mp4"):
                try:
                    if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} old video clips")
    
    def start_scheduler(self):
        """Start the monitoring scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.setup_directories()
        self.setup_scheduled_jobs()
        
        # Schedule daily cleanup at 2 AM
        schedule.every().day.at("02:00").do(self.cleanup_old_files)
        
        self.is_running = True
        logger.info("Medicine monitoring scheduler started")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            self.send_error_notification(f"Scheduler crashed: {str(e)}")
        finally:
            self.stop_scheduler()
    
    def stop_scheduler(self):
        """Stop the monitoring scheduler"""
        self.is_running = False
        if hasattr(self.detector, 'release'):
            self.detector.release()
        logger.info("Medicine monitoring scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            "is_running": self.is_running,
            "scheduled_jobs": len(schedule.jobs),
            "scheduled_times": self.config.get("scheduled_times", []),
            "recent_history": self.monitoring_history[-5:],  # Last 5 sessions
            "config_file": self.config_file,
            "email_notifications_enabled": self.email_config.get("enabled", False),
            "next_scheduled_run": str(schedule.next_run()) if schedule.jobs else None
        }
    
    def add_schedule(self, time_str: str, duration: int = 60, enabled: bool = True):
        """Add a new scheduled monitoring time"""
        # Validate time format
        try:
            datetime.strptime(time_str, "%H:%M")
        except ValueError:
            raise ValueError("Time must be in HH:MM format")
        
        # Check if schedule already exists
        existing_times = [s["time"] for s in self.config["scheduled_times"]]
        if time_str in existing_times:
            raise ValueError(f"Schedule for {time_str} already exists")
        
        new_schedule = {
            "time": time_str,
            "duration": duration,
            "enabled": enabled
        }
        
        self.config["scheduled_times"].append(new_schedule)
        self.save_config()
        
        if self.is_running:
            self.setup_scheduled_jobs()
        
        logger.info(f"Added new schedule: {time_str} for {duration}s")
    
    def remove_schedule(self, time_str: str):
        """Remove a scheduled monitoring time"""
        original_count = len(self.config["scheduled_times"])
        self.config["scheduled_times"] = [
            s for s in self.config["scheduled_times"] 
            if s["time"] != time_str
        ]
        
        if len(self.config["scheduled_times"]) < original_count:
            self.save_config()
            if self.is_running:
                self.setup_scheduled_jobs()
            logger.info(f"Removed schedule: {time_str}")
            return True
        else:
            logger.warning(f"Schedule not found: {time_str}")
            return False
    
    def update_schedule(self, time_str: str, duration: int = None, enabled: bool = None):
        """Update an existing schedule"""
        for schedule_config in self.config["scheduled_times"]:
            if schedule_config["time"] == time_str:
                if duration is not None:
                    schedule_config["duration"] = duration
                if enabled is not None:
                    schedule_config["enabled"] = enabled
                
                self.save_config()
                if self.is_running:
                    self.setup_scheduled_jobs()
                
                logger.info(f"Updated schedule: {time_str}")
                return True
        
        logger.warning(f"Schedule not found for update: {time_str}")
        return False
    
    def get_monitoring_results(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get monitoring results from files"""
        results_dir = Path(self.config.get("storage", {}).get("results_directory", "monitoring_results"))
        results = []
        
        if not results_dir.exists():
            return results
        
        # Get all result files, sorted by modification time (newest first)
        result_files = sorted(
            results_dir.glob("monitoring_result_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if limit:
            result_files = result_files[:limit]
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to load result file {file_path}: {e}")
        
        return results


def main():
    """Main function to run the scheduler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medicine Monitoring Scheduler Service")
    parser.add_argument("--config", default="monitoring_config.json", 
                       help="Configuration file path")
    parser.add_argument("--add-schedule", nargs=2, metavar=("TIME", "DURATION"),
                       help="Add a schedule (TIME in HH:MM format, DURATION in seconds)")
    parser.add_argument("--remove-schedule", metavar="TIME",
                       help="Remove a schedule (TIME in HH:MM format)")
    parser.add_argument("--status", action="store_true",
                       help="Show current status")
    parser.add_argument("--test", action="store_true",
                       help="Run a test monitoring session")
    parser.add_argument("--results", type=int, metavar="LIMIT",
                       help="Show recent monitoring results (optional limit)")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon service")
    
    args = parser.parse_args()
    
    scheduler = MedicineMonitoringScheduler(args.config)
    
    if args.add_schedule:
        time_str, duration_str = args.add_schedule
        try:
            duration = int(duration_str)
            scheduler.add_schedule(time_str, duration)
            print(f"✅ Added schedule: {time_str} for {duration} seconds")
        except ValueError as e:
            print(f"❌ Error: {e}")
            return
        except Exception as e:
            print(f"❌ Failed to add schedule: {e}")
            return
    
    elif args.remove_schedule:
        if scheduler.remove_schedule(args.remove_schedule):
            print(f"✅ Removed schedule: {args.remove_schedule}")
        else:
            print(f"❌ Schedule not found: {args.remove_schedule}")
    
    elif args.status:
        status = scheduler.get_status()
        print("📊 Medicine Monitoring Scheduler Status:")
        print("=" * 50)
        print(json.dumps(status, indent=2, default=str))
    
    elif args.results is not None:
        results = scheduler.get_monitoring_results(args.results)
        print(f"📈 Recent Monitoring Results (showing {len(results)} results):")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            status = "✅ DETECTED" if result.get("medicine_intake_detected") else "❌ NOT DETECTED"
            print(f"{i}. {result.get('monitoring_date', 'Unknown date')} - {status}")
            print(f"   Duration: {result.get('duration_seconds', 0)}s, "
                  f"Detections: {result.get('total_detections', 0)}")
            if result.get("scheduled_time"):
                print(f"   Scheduled time: {result['scheduled_time']}")
            print()
    
    elif args.test:
        print("🧪 Running test monitoring session (30 seconds)...")
        try:
            result = scheduler.run_monitoring_session(30)
            print("✅ Test completed!")
            print("Results:")
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    else:
        # Start the scheduler service
        print("🚀 Starting Medicine Monitoring Scheduler Service...")
        print("Configuration:", args.config)
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            scheduler.start_scheduler()
        except KeyboardInterrupt:
            print("\n🛑 Stopping scheduler...")
            scheduler.stop_scheduler()
            print("✅ Scheduler stopped successfully")


if __name__ == "__main__":
    main()