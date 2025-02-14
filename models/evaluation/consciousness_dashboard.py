
from flask import Flask, render_template, jsonify
import threading
import time
import random  # For demo; replace with your real metrics fetching

app = Flask(__name__)
consciousness_history = []

def fetch_metrics():
    """
    Demo background function that simulates updating
    consciousness metrics periodically.
    Replace this with real calls to your ConsciousnessMonitor.
    """
    while True:
        # Example random metrics
        consciousness_score = random.uniform(0.0, 1.0)
        memory_coherence = random.uniform(0.0, 1.0)
        global_workspace = random.uniform(0.0, 1.0)
        consciousness_history.append({
            "score": consciousness_score,
            "memory_coherence": memory_coherence,
            "global_workspace": global_workspace,
            "timestamp": time.time()
        })
        time.sleep(2)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/metrics")
def get_metrics():
    return jsonify(consciousness_history[-50:])  # Last 50 points

def run_dashboard():
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    # Start background thread for data collection
    metrics_thread = threading.Thread(target=fetch_metrics, daemon=True)
    metrics_thread.start()

    # Start Flask server
    run_dashboard()