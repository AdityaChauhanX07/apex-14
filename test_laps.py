import socket, struct, time

tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind(('0.0.0.0', 20778))
rx.settimeout(2.0)

print("Driving laps on Silverstone (moderate throttle)...")
seq = 0
last_lap = 1
rx.setblocking(False)

t_start = time.time()
while time.time() - t_start < 300:  # 5 minutes max
    seq += 1
    t = time.time() - t_start
    # Gentle steering corrections + moderate throttle
    steer = 0.0
    throttle = 0.7
    packet = struct.pack('<fffII', steer, throttle, 0.0, 3, seq)
    tx.sendto(packet, ('127.0.0.1', 20777))
    
    while True:
        try:
            data, _ = rx.recvfrom(256)
            # Parse with new 208-byte format
            speed = struct.unpack_from('<d', data, 48)[0]
            lap = struct.unpack_from('<I', data, 168)[0]
            lap_time = struct.unpack_from('<d', data, 176)[0]  # adjusted for new fields
            track_dist = struct.unpack_from('<d', data, 192)[0]
            track_off = struct.unpack_from('<d', data, 200)[0]
            
            if lap != last_lap:
                print(f"  LAP {lap} at t={t:.1f}s, lap_time={lap_time:.2f}s")
                last_lap = lap
            
        except BlockingIOError:
            break
    
    # Progress every 30s
    if seq % 3000 == 0:
        print(f"  {t:.0f}s: speed={speed:.1f} m/s, track_dist={track_dist:.0f}m, offset={track_off:.1f}m, lap={lap}")
    
    time.sleep(0.01)

print("Done")
tx.close()
rx.close()