#!/bin/bash

# Start the backend server
cd backend
export FLASK_ENV=development
python app.py &

# Wait for backend to start
sleep 3

# Start the frontend
cd ../intelliquery
npm run dev