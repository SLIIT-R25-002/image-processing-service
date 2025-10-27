from pyngrok import ngrok
import threading
import time
from app import app

def run_app():
    """Run Flask app in a background thread"""
    app.run(host="0.0.0.0", port=5000)

def start_server():
    """Start the Flask server with ngrok tunnel"""
    print("🚀 Starting Flask app and preloading models...")
    flask_thread = threading.Thread(target=run_app)
    flask_thread.daemon = True
    flask_thread.start()
    print("⏱️ Waiting for server and models to initialize...")
    time.sleep(10)  # Give more time for model loading

    # Create the public URL
    try:
        public_url = ngrok.connect(5000)
        print("====================================================================")
        print(f"✅ Your Flask API is now live!")
        print(f"🌍 Public URL: {public_url}")
        print("====================================================================")
        print("You can now use this URL in a tool like Postman to test your API.")
        print("Press Ctrl+C to stop the server...")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        ngrok.disconnect(public_url)
    except Exception as e:
        print(f"❌ Error starting ngrok: {e}")

if __name__ == "__main__":
    run_app()
