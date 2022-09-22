from picamera2 import Picamera2
import subprocess

def cam2():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    picam2.start()
    p = subprocess.check_output("cd static/imgs/; ls -l | grep ^- | wc -l; cd ../..",shell=True, encoding='utf-8')
    p =str(p)
    p = p[:len(p)-1]
    print(p)
    picam2.capture_file(f"./static/imgs/pic{p}.jpg")
    pic_path = f"./static/imgs/pic{p}.jpg"
    picam2.close()
    return pic_path # ./static/imgs/pic{p}.jpg #, p