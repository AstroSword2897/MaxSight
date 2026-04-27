#!/bin/bash
# MaxSight Product Simulator Startup Script

export MAXSIGHT_RUNTIME="${MAXSIGHT_RUNTIME:-simulator}"

echo "=========================================="
echo "MaxSight Product Simulator"
echo "=========================================="
echo "MAXSIGHT_RUNTIME=$MAXSIGHT_RUNTIME"
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
PORT_HINT="${MAXSIGHT_PORT:-8002}"
echo "📍 Use the URL printed when the server starts (default port ${PORT_HINT}, same as DEFAULT_SIMULATOR_PORT in tools/simulation/config.py)"
echo "   Override: MAXSIGHT_PORT=9000  |  fail if taken: MAXSIGHT_STRICT_PORT=1"
echo "   Behind nginx/ALB: MAXSIGHT_BEHIND_PROXY=1 (uses X-Forwarded-Host / port for routing labels)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 web_simulator.py

