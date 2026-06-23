import subprocess
import threading

def capture_stereo_images(camera_id, filename):
    subprocess.run([
        'rpicam-still',
        '--camera', str(camera_id),
        '--nopreview',
        '--width', '1640',
        '--height', '1232',
        '-o', filename
    ])
    
    # Capture the both cameras simultaneously
    t0 = threading.Thread(target=capture_stereo_images, args=(0, 'left2.jpg'))
    t1 = threading.Thread(target=capture_stereo_images, args=(1, 'right2.jpg'))
    
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    
    print("Stereo images captured")