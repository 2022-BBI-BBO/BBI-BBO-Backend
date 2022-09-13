from picamera2 import Picamera2
import subprocess

comand = 'cd static/imgs/; ls -l | grep ^- | wc -l; cd ../..'

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
    return p , pic_path

# picam2.start_preview(Preview.DRM)
    # t1 = subprocess.check_output("cd imgs/; ls -l | grep ^- | wc -l; cd ../",shell=True, encoding='utf-8')
    # t2 = subprocess.call("cd imgs/; ls -l | grep ^- | wc -l; cd ../",shell=True)
    # t1 = str(t1)
    # t1 = t1[:len(t1)-3]
    # print(f"t1 : {t1} t2 : {t2}")
    # p = subprocess.call("cd ./static/imgs/; ls -l | grep ^- | wc -l; cd ../",shell=True)