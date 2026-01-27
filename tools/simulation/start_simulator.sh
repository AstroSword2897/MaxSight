#!/bin/bash
# MaxSight Product Simulator Startup Script

echo "=========================================="
echo "MaxSight Product Simulator"
echo "=========================================="
echo ""

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask not found. Installing..."
    pip install flask flask-cors
fi

# Change to simulator directory
cd "$(dirname "$0")"

# Start the simulator
echo "🚀 Starting MaxSight Simulator..."
echo ""
echo "📍 Access the simulator at: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 web_simulator.py

