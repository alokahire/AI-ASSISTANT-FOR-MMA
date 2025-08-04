import cv2
import numpy as np
import time

# Configuration
PLAYERS = {
    "RED": {
        "video": "Player1.mp4",
        "color": {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},
        "text_color": (0, 0, 255)  # BGR - Red
    },
    "WHITE": {
        "video": "Player2.mp4",
        "color": {"lower": np.array([0, 0, 200]), "upper": np.array([180, 30, 255])},
        "text_color": (255, 255, 255)  # BGR - White
    }
}

# Initialize video captures
caps = {}
for name, data in PLAYERS.items():
    cap = cv2.VideoCapture(data["video"])
    if not cap.isOpened():
        print(f"Error opening {name} player video '{data['video']}'! Skipping this player.")
        continue
    caps[name] = cap

if not caps:
    print("No videos loaded successfully. Exiting program.")
    exit()

# UFC-style display settings
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 1
COUNTER_POS = (20, 40)  # (x,y) position for counters
TIMER_POS = (20, 80)    # Position for round timer

# Strike counters
counters = {name: {"punches": 0, "kicks": 0} for name in caps}

def detect_player(frame, color_range):
    """Detect player presence using color thresholding"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
    return cv2.countNonZero(mask) > 500  # Detection threshold

def draw_ufc_overlay(frame, player_name, punch_count, kick_count, elapsed_time):
    """Draw UFC-style information overlay"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, player_name, 
                (COUNTER_POS[0], COUNTER_POS[1]-25), 
                FONT, FONT_SCALE, PLAYERS[player_name]["text_color"], FONT_THICKNESS)
    
    cv2.putText(frame, f"PUNCHES: {punch_count}", 
                COUNTER_POS, FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS)
    
    cv2.putText(frame, f"KICKS: {kick_count}", 
                (COUNTER_POS[0], COUNTER_POS[1]+30), FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS)
    
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", 
                TIMER_POS, FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS)
    return frame

# Main loop
start_time = time.time()
try:
    while True:
        current_time = time.time() - start_time
        
        for player_name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or error reading frame for {player_name}.")
                continue
                
            if detect_player(frame, PLAYERS[player_name]["color"]):
                counters[player_name]["punches"] += 1
                if counters[player_name]["punches"] % 5 == 0:
                    counters[player_name]["kicks"] += 1
            
            frame = draw_ufc_overlay(
                frame, 
                player_name,
                counters[player_name]["punches"],
                counters[player_name]["kicks"],
                current_time
            )
            
            cv2.imshow(f"{player_name} PLAYER", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting on user request.")
            break

finally:
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
