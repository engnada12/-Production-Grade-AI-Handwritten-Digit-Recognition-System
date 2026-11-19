"""
Handwritten Digit Recognizer - Production REST API & Web Server
Provides high-performance RESTful endpoints for digit recognition (1-9)
Works with zero external web dependencies via Python's standard library,
with full compatibility for FastAPI/Uvicorn when deployed in Docker.
"""

import os
import sys
import json
import time
import base64
from email import message_from_bytes
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

# Fix Windows console encoding for cp1256 / ANSI
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure local modules are accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model_handler import predict_digit, load_model, get_model_path

SERVER_START_TIME = time.time()
STATIC_DIR = os.path.join(current_dir, "static")
FEEDBACK_DIR = os.path.join(current_dir, "data", "feedback")
os.makedirs(FEEDBACK_DIR, exist_ok=True)


class DigitRecognizerHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler serving static frontend and RESTful AI API."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)
        
    def _send_json_response(self, data, status_code=200):
        """Helper to send standardized JSON response with CORS headers."""
        response_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(response_bytes)
        
    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        
    def do_GET(self):
        """Route GET requests: APIs vs Static files."""
        if self.path == "/api/health":
            self.handle_health()
        elif self.path == "/api/model-info":
            self.handle_model_info()
        elif self.path.startswith("/api/"):
            self._send_json_response({"status": "error", "message": "Endpoint not found"}, 404)
        else:
            # Serve Frontend static assets
            super().do_GET()
            
    def do_POST(self):
        """Route POST requests: Predict & Feedback APIs."""
        if self.path == "/api/predict":
            self.handle_predict()
        elif self.path == "/api/feedback":
            self.handle_feedback()
        else:
            self._send_json_response({"status": "error", "message": "Endpoint not found"}, 404)
            
    def handle_health(self):
        """Health-check endpoint for load balancers & monitoring."""
        try:
            _, model_type = load_model()
            uptime_seconds = round(time.time() - SERVER_START_TIME, 1)
            self._send_json_response({
                "status": "healthy",
                "uptime_seconds": uptime_seconds,
                "active_model": model_type,
                "version": "2.0.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
        except Exception as e:
            self._send_json_response({
                "status": "unhealthy",
                "error": str(e)
            }, 503)
            
    def handle_model_info(self):
        """Returns model capabilities, target benchmarks, and architecture specs."""
        try:
            model_path, model_type = get_model_path()
            self._send_json_response({
                "status": "success",
                "model_name": "Handwritten Digits Recognizer (Digits 1-9)",
                "active_model": model_type,
                "model_file": os.path.basename(model_path) if model_path else "None",
                "target_digits": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "performance_benchmarks": {
                    "target_accuracy": ">= 98.0%",
                    "target_latency_ms": "< 200 ms",
                    "typical_cpu_latency_ms": "~15-35 ms",
                    "input_resolution": "28x28 grayscale (with auto-centering & aspect-ratio normalization)"
                },
                "supported_modalities": ["Upload Image (PNG/JPEG)", "Live Camera Feed", "Interactive Drawing Canvas"]
            })
        except Exception as e:
            self._send_json_response({"status": "error", "message": str(e)}, 500)
            
    def handle_predict(self):
        """Handles digit recognition request (supports JSON base64 & multipart form)."""
        try:
            content_type = self.headers.get("Content-Type", "")
            image_data = None
            
            if "application/json" in content_type:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                req_json = json.loads(body.decode("utf-8"))
                image_data = req_json.get("image")
            elif "multipart/form-data" in content_type:
                content_len = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_len)
                msg_data = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + raw_body
                msg = message_from_bytes(msg_data)
                for part in msg.walk():
                    if part.get_content_maintype() == 'image' or part.get_filename():
                        image_data = part.get_payload(decode=True)
                        break
            else:
                # Direct binary or raw base64 body
                content_len = int(self.headers.get("Content-Length", 0))
                image_data = self.rfile.read(content_len)
                
            if not image_data:
                self._send_json_response({
                    "status": "error",
                    "message": "Missing image in request. Please provide 'image' as base64 or multipart file."
                }, 400)
                return
                
            # Perform inference
            result = predict_digit(image_data, target_digits_only=True)
            self._send_json_response(result, 200)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json_response({
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }, 500)
            
    def handle_feedback(self):
        """Saves user feedback for continuous improvement / Active Learning."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            feedback_data = json.loads(body.decode("utf-8"))
            
            actual_digit = feedback_data.get("actual_digit")
            predicted_digit = feedback_data.get("predicted_digit")
            image_b64 = feedback_data.get("image")
            
            if actual_digit is not None and image_b64:
                timestamp = int(time.time() * 1000)
                filename = f"feedback_pred{predicted_digit}_actual{actual_digit}_{timestamp}.png"
                filepath = os.path.join(FEEDBACK_DIR, filename)
                
                # Save image
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(image_b64))
                    
                meta_path = os.path.join(FEEDBACK_DIR, f"meta_{timestamp}.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "predicted": predicted_digit,
                        "actual": actual_digit,
                        "timestamp": timestamp,
                        "client_ip": self.client_address[0]
                    }, f, indent=2)
                    
            self._send_json_response({
                "status": "success",
                "message": "Feedback successfully recorded. Thank you for contributing to model improvement!"
            }, 200)
        except Exception as e:
            self._send_json_response({
                "status": "error",
                "message": f"Failed to record feedback: {str(e)}"
            }, 500)


def run_server(port=8080):
    """Starts the HTTP server."""
    # Pre-warm model in memory
    print("[Server] Pre-warming machine learning model...")
    try:
        load_model()
    except Exception as e:
        print(f"[Server Warning] Model loading deferred: {e}")
        
    handler = DigitRecognizerHandler
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print("\n=======================================================")
        print(f"[*] AI Handwritten Digit Recognizer Server is running!")
        print(f"[*] Local URL:   http://localhost:{port}")
        print(f"[*] Health API:  http://localhost:{port}/api/health")
        print(f"[*] Model Info:  http://localhost:{port}/api/model-info")
        print("=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Server] Shutting down gracefully...")
            httpd.server_close()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port=port)
